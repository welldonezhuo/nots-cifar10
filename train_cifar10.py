import argparse
import os
import torch
from diffusers import FluxPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--output", type=str, default="flux-dev.png")
args = parser.parse_args()

hf_token = os.environ.get("HF_TOKEN")

if hf_token:
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        token=hf_token
    )
else:
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )

pipe.enable_model_cpu_offload()

image = pipe(
    args.prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

image.save(args.output)
print(f"Saved image to {args.output}")