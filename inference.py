import os
import torch
import numpy as np
from diffusers import UNet2DModel, AutoencoderKL
from diffusers import PNDMPipeline
from accelerate import Accelerator
from tqdm.auto import tqdm
from diffusers import DDPMPipeline
from PIL import Image
from fid_score.fid_score import FidScore

device = 'cuda'
data_size = 70000
vae = AutoencoderKL.from_pretrained('vae')
vae = vae.to(device)
ffhq_path = '/home/anna/ml-hdd/ffhq512'

def randn_tensor(
    shape,
    generator = None,
    device = None,
    dtype = None,
    layout = None,
):
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

def decode_img_latents(latents):
    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        imgs = vae.decode(latents)[0]

    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in imgs]
    return pil_images

def make_latents(pipeline,
                batch_size: int = 1,
                num_inference_steps: int = 50,
                generator = None,
               ):
    image = randn_tensor(
            (batch_size, pipeline.unet.config.in_channels, pipeline.unet.config.sample_size, pipeline.unet.config.sample_size),
            generator=generator,
            device=pipeline.device,
        )

    pipeline.scheduler.set_timesteps(num_inference_steps)
    for t in pipeline.progress_bar(pipeline.scheduler.timesteps):
        model_output = pipeline.unet(image, t).sample
        image = pipeline.scheduler.step(model_output, t, image).prev_sample
            
    return image

def main():
    pipeline_path = "/home/anna/ml-hdd/Diffusion/pipe"
    pipeline = PNDMPipeline.from_pretrained(pipeline_path, use_safetensors=True)

    for i in tqdm(range(0, data_size)):
        pipeline_output = make_latents(pipeline, num_inference_steps=20).to(device)
        res = decode_img_latents(pipeline_output)[0]
        res.save(f"decoded/{i:05d}.png")
                
    fid = FidScore(['decoded', ffhq_path], device, batch_size)
    print("FID is", fid.calculate_fid_score())
    
    
if __name__ == "__main__":
    main()
