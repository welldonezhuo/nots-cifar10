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
hf_home = os.environ.get("HF_HOME")

print(f"HF_TOKEN exists: {hf_token is not None}", flush=True)
print(f"HF_HOME: {hf_home}", flush=True)

try:
    if not hf_home:
        raise RuntimeError("HF_HOME is not set")

    os.makedirs(hf_home, exist_ok=True)

    print("Loading pipeline...", flush=True)
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        token=hf_token,
        cache_dir=hf_home
    )

    print("Pipeline loaded successfully", flush=True)

    if torch.cuda.is_available():
        print("Moving pipeline to CUDA...", flush=True)
        pipe = pipe.to("cuda")
    else:
        print("CUDA not available, enabling CPU offload...", flush=True)
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

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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