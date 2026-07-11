# Prefill server: GLM-5.2 MXFP4-FP8, single replica PURE TP4 (no DP, no EP), gfx950.
# MoE runs tensor-parallel (experts sharded on intermediate dim across the 4 GPUs,
# TP all-reduce, no expert-parallel all2all). Run two of these (GPUs 0-3 and 4-7)
# behind the nginx LB to compare against the TP4+EP4 variant.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

export MORI_GPU_ARCHS=gfx950
rm -rf /root/.mori/jit/gfx942_* 2>/dev/null || true

VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=${FUSION_SHARED:-1} \
VLLM_ROCM_FP8_BLOCKSCALE_PRESHUFFLE=1 \
VLLM_USE_AITER_INDEXER_TOPK_FAST=0 \
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=0 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
VLLM_ROCM_FUSED_MLA_ROPE_CACHE=1 \
vllm serve /workspace/models/GLM-5.2-MXFP4-FP8 \
  --port ${PORT:-8000} \
  --no-enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --tensor-parallel-size 4 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 8192 \
  --max-model-len 103424 \
  --max-num-seqs 16 \
  --linear-backend aiter \
  --moe-backend aiter \
  --compilation-config '{"cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [4, 8, 16]}'
