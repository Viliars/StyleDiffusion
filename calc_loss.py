import os
import math
import torch
import logging
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import diffusers
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import EMAModel, compute_snr
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed as accelerate_set_seed
from tqdm.auto import tqdm
from config import config
from dataset import LatentFFHQ


logger = get_logger(__name__, log_level="INFO")


def create_unet():
    if config.pretrained is None:
        model = UNet2DModel(
            sample_size=32,
            in_channels=4,
            out_channels=4,
            center_input_sample=False,
            time_embedding_type='positional',
            freq_shift=0,
            flip_sin_to_cos=True,
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels=(160, 320, 640, 1280),
            layers_per_block=2,
            mid_block_scale_factor=1,
            downsample_padding=1,
            norm_eps=1e-05,
            norm_num_groups=32,
            attention_head_dim=8,
            act_fn="silu"
        )
        logger.info("The UNet2DModel was created from scratch!")
    else:
        model = UNet2DModel.from_pretrained(config.pretrained, use_safetensors=True)
        logger.info("The UNet2DModel was loaded from the checkpoint!")

    return model


def main():
    # настраиваем accelerate
    logging_dir = os.path.join(config.output_dir, config.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
    )

    # настраиваем логгирование
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # фиксируем seed
    accelerate_set_seed(config.seed)

    # создаем output папку, если ее нет
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

    # создаем scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.ddpm_num_steps,
        beta_schedule=config.ddpm_beta_schedule,
        prediction_type=config.prediction_type,
    )

    # создаем модель
    unet = create_unet()
    
    unet.eval()

    if config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # создаем датасет и лоадер
    dataset = LatentFFHQ(root="latent32")
    dataloader = DataLoader(dataset,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_num_workers,
        shuffle=True
    )

    num_update_steps_per_epoch = math.ceil(len(dataloader) / config.gradient_accumulation_steps)
    config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch

    unet, dataloader = accelerator.prepare(
        unet, dataloader
    )

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        config.mixed_precision = accelerator.mixed_precision

    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    logger.info("***** Running calculating loss *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    global_step = 0

    train_loss = 0.0
    progress_bar = tqdm(total=(num_update_steps_per_epoch // accelerator.num_processes), disable=not accelerator.is_local_main_process)

    for step, latents in enumerate(dataloader):
        with accelerator.accumulate(unet):
            latents = latents.to(weight_dtype)
            noise = torch.randn_like(latents)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)

            model_pred = unet(noisy_latents, timesteps, return_dict=False)[0]

            if config.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                    dim=1
                )[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
            train_loss += avg_loss.item() / config.gradient_accumulation_steps

        
        if accelerator.sync_gradients:

            progress_bar.update(1)
            progress_bar.set_description(f"LR {lr_scheduler.get_lr()}; Epoch {epoch}")
            global_step += 1
            accelerator.log({"train_loss": train_loss}, step=global_step)
            train_loss = 0.0

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
