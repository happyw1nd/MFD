import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from diffusers import SanaPipeline
import diffusers
from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    SanaPipeline,
    SanaTransformer2DModel,
)

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    variant="bf16",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
pipe.load_lora_weights("outputs/checkpoint-1000/pytorch_lora_weights.safetensors") # TODO: change the path to your checkpoint

pipe.vae.to(torch.float32)
pipe.text_encoder.to(torch.bfloat16)

negatival_prompt = 'blurry, bad anatomy, bad hands, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
prompt = 'Fantsy landscape with a river and a small village on top of a rock arch. Digital illustration. Stock Photo'
image = pipe(
    prompt=prompt,
    negative_prompt=negatival_prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device="cuda").manual_seed(42),
)[0]

print(pipe.scheduler.timesteps)

image[0].save("sana.png")
