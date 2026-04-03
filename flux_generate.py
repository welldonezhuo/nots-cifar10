import argparse
import torch
from diffusers import FluxPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--output", type=str, default="flux-dev.png")
args = parser.parse_args()

print("Script started", flush=True)
print(f"Output file: {args.output}", flush=True)

print("Loading pipeline...", flush=True)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)

print("Pipeline loaded", flush=True)

pipe.enable_model_cpu_offload()
print("CPU offload enabled", flush=True)

print("Starting generation...", flush=True)
image = pipe(
    args.prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

print("Generation finished", flush=True)

image.save(args.output)
print(f"Saved image to {args.output}", flush=True)