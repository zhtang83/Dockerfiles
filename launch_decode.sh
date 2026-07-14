VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
VLLM_ROCM_FP8_BLOCKSCALE_PRESHUFFLE=1 \
VLLM_USE_AITER_INDEXER_TOPK_FAST=0 \
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=0 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
VLLM_ROCM_FUSED_MLA_ROPE_CACHE=1 \
vllm serve /home/zhtang/GLM-5.2-MXFP4-FP8 \
  --port ${PORT:-8000} \
  --no-enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --tensor-parallel-size 8 \
  --block-size 64 \
  --speculative-config.method mtp --speculative-config.num_speculative_tokens 5 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 8192 \
  --max-model-len 122880 \
  --max-num-seqs 64 \
  --linear-backend aiter \
  --moe-backend aiter
