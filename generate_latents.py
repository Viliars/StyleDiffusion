import os
import torch
import numpy as np
from tqdm.auto import tqdm
from diffusers import DDPMPipeline
from PIL import Image
from fid_score.fid_score import FidScore
from diffusers.utils.torch_utils import randn_tensor

device = 'cuda'
data_size = 70000
batch_size = 2

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
    pipeline_path = "/home/anna/ml-hdd/Diffusion/model-epoch011"
    pipeline = DDPMPipeline.from_pretrained(pipeline_path, use_safetensors=True)
    pipeline = pipeline.to(device)

    for i in tqdm(range(0,data_size, batch_size)):
        with torch.no_grad():
            pipeline_output = make_latents(pipeline, batch_size=batch_size, num_inference_steps=50)
            pipeline_output = pipeline_output.cpu().numpy()
            for j in range(pipeline_output.shape[0]):
                k = i + j

                np.save(f"latents/{k:05d}.npy", pipeline_output[j])


if __name__ == "__main__":
    main()
