import os
import torch
import numpy as np
from diffusers import AutoencoderKL
from tqdm.auto import tqdm
from diffusers import DDPMPipeline
from PIL import Image
from fid_score.fid_score import FidScore
from diffusers.utils.torch_utils import randn_tensor
device = 'cuda'
data_size = 70000
vae = AutoencoderKL.from_pretrained('vae')
vae = vae.to(device)
ffhq_path = '/home/anna/ml-hdd/ffhq512'

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
    pipeline_path = "/home/anna/ml-hdd/Diffusion/model-epoch011"
    pipeline = DDPMPipeline.from_pretrained(pipeline_path, use_safetensors=True)
    pipeline = pipeline.to(device)
    batch_size=10

    for i in tqdm(range(22000, data_size, batch_size)):
        with torch.no_grad():
            pipeline_output = make_latents(pipeline, batch_size=batch_size, num_inference_steps=50).to(device)
            res = decode_img_latents(pipeline_output)
            for j in range(len(res)):
                k = i + j
                res[j].save(f"decoded/{k:05d}.png")
                
    fid = FidScore(['decoded', ffhq_path], device, batch_size)
    print("FID is", fid.calculate_fid_score())
    
    
if __name__ == "__main__":
    main()
