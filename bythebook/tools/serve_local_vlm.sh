#!/usr/bin/env bash
# Local vision-language model for bythebook/ (--vlm local). OpenAI-compatible API on port 8000.
# The 32B AWQ weights take about 23 GB of the 32 GB card with their runtime buffers; the
# context is kept at 10k tokens so that a 16-bit KV cache fits and about 5 GB stay free
# for the perception models. LocalVLM sends the manual at 80 dpi and crops at 896 px to
# stay inside that context.
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=~/miniforge3/envs/vllm
exec ~/miniforge3/envs/vllm/bin/vllm serve QuantTrio/Qwen3-VL-32B-Instruct-AWQ \
  --served-model-name Qwen3-VL-32B --port 8000 --host 127.0.0.1 \
  --max-model-len 10240 --gpu-memory-utilization 0.80 \
  --limit-mm-per-prompt '{"image": 16}' --max-num-seqs 2
