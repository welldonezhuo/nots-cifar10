import argparse
import os
import traceback
import torch
from diffusers import FluxPipeline

print("=== SCRIPT STARTED ===", flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--output", type=str, default="flux-dev.png")
args = parser.parse_args()

hf_token = os.environ.get("HF_TOKEN")
print(f"HF_TOKEN exists: {hf_token is not None}", flush=True)
print(f"HF_HOME: {os.environ.get('HF_HOME')}", flush=True)

try:
    print("Loading pipeline...", flush=True)
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        token=hf_token
    )

    print("Pipeline loaded successfully", flush=True)
    print("Enabling CPU offload...", flush=True)
    pipe.enable_model_cpu_offload()

    print("Generating image...", flush=True)
    image = pipe(
        args.prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]

    print("Saving image...", flush=True)
    image.save(args.output)
    print(f"Saved image to {args.output}", flush=True)
    print("=== SCRIPT FINISHED ===", flush=True)

except BaseException as e:
    print("=== PYTHON EXCEPTION ===", flush=True)
    print("Type:", type(e).__name__, flush=True)
    print("Repr:", repr(e), flush=True)
    traceback.print_exc()
    raise