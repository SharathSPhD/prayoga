# prayoga GPU environment (DGX Spark / GB10)

Verified working recipe for Tier-2 mech-interp on the GB10 (Grace-Blackwell,
aarch64, sm_121, CUDA 13, driver 580).

## What works (verified 2026-06-25)
- Base: `nvcr.io/nvidia/vllm:26.02-py3` → torch `2.11.0a0+...nv26.02`,
  `cuda=True`, device `NVIDIA GB10`, capability `(12, 1)`.
- `transformers 4.57.5` (preinstalled), `+ scikit-learn einops accelerate`.
- `google/gemma-2-2b-it` loads on GPU (~68 s cold) and refuses harmful prompts.

## The transformer-lens trap (do not repeat)
`pip install transformer-lens` pulls a PyPI **CPU** torch 2.7.1 + transformers
v5, **clobbering** the NVIDIA CUDA torch → `Torch not compiled with CUDA enabled`.
prayoga therefore implements Axis-A interp on **raw `transformers` + forward
hooks** (`prayoga.lm.hf_model`). SAE / circuit-tracer, when needed, run in a
separate isolated env/container so they cannot break this one.

## Run

```bash
# persistent container (repo + HF cache mounted; HF token enables gated models)
docker run -d --name prayoga-gpu --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $PWD:/workspace/prayoga \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace/prayoga nvcr.io/nvidia/vllm:26.02-py3 sleep infinity
docker exec prayoga-gpu pip install --no-cache-dir scikit-learn einops accelerate

# run an experiment
docker exec prayoga-gpu python -m prayoga.axis_a.run_direction --model gemma-2-2b-it
```

Or build the pinned image: `docker build -f docker/Dockerfile.dgx-spark -t prayoga-gpu .`
