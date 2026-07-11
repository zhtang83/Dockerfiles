# Prefill server: GLM-5.2 MXFP4-FP8, single replica TP4 + EP4 (NO data parallel), gfx950.
# Purpose: MoE expert-parallel group is confined to the 4 GPUs of this one instance
#   (ep_size = tp_size = 4), so it does NOT span a DP boundary. Run two of these
#   (GPUs 0-3 and 4-7) behind an external load balancer to get 2 independent replicas
#   with zero cross-replica MoE all2all coupling.
# Same quant / kernels / flags as vllm_glm_fp4_prefill.sh, minus --data-parallel-size.

# Pin to first 4 GPUs. For the second instance use CUDA_VISIBLE_DEVICES=4,5,6,7 and --port 8001.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# MoRI JITs only the first of MORI_GPU_ARCHS (gfx942 from /etc/environment),
# invalid on gfx950. Force gfx950 only.
export MORI_GPU_ARCHS=gfx950
rm -rf /root/.mori/jit/gfx942_* 2>/dev/null || true

VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 \
VLLM_DISABLE_SP_MOE=0 \
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
  --enable-expert-parallel \
  --all2all-backend mori_high_throughput \
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
