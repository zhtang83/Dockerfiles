# GLM-5.2 MXFP4-FP8 — Prefill serving (ROCm gfx950 / 8×MI355X)

Prefill-optimized deployment. To avoid the cross-DP MoE coupling that makes a single
`DP2×TP4` engine slow (bimodal ~4.8s TTFT), prefill runs as **two independent TP4
processes behind an nginx `least_conn` load balancer**. This fully decouples the two
replicas — a 60k-token prefill lands on one replica while the other serves the next
request in parallel.

## Topology (both variants)

```
                         ┌─ Replica A : GPUs 0-3 -> 127.0.0.1:8001 ─┐
client ─▶ nginx :8000 ───┤  (least_conn 1+1)                        │
                         └─ Replica B : GPUs 4-7 -> 127.0.0.1:8002 ─┘
```
Clients hit **`http://127.0.0.1:8000`** (OpenAI-compatible), same as a single vLLM.

## Two variants

| Variant | Script | MoE mode | 60k/conc2 TTFT |
|---|---|---|---|
| **TP4 + EP4** (recommended) | `start_2tp4_lb.sh` | expert-parallel within each 4-GPU replica (all2all confined to 4 GPUs) | **3.11s** |
| Pure TP4 | `start_2tp4_pure_lb.sh` | tensor-parallel MoE (no EP, no all2all) | 3.25s |

Both: shared expert active, quickreduce INT4, fp8 KV, PIECEWISE cudagraph, no bimodal
jitter. EP4 is ~4% faster; pure TP4 is simpler (no MoRI/EP path).

## Usage

```bash
# TP4+EP4 (recommended)
bash start_2tp4_lb.sh              # start 2 replicas + LB, wait until ready
bash start_2tp4_lb.sh stop         # stop everything

# Pure TP4
bash start_2tp4_pure_lb.sh         # FUSION_SHARED=1 (shared expert fused) by default
FUSION_SHARED=0 bash start_2tp4_pure_lb.sh   # standalone shared-expert variant
bash start_2tp4_pure_lb.sh stop
```

Point the client at `http://127.0.0.1:8000`, e.g.
`vllm bench serve --base-url http://127.0.0.1:8000 ...`.

## Requirements
- `nginx` installed (`apt-get install -y nginx`). `least_conn` is mandatory — default
  round-robin/ip_hash can pile both concurrent prefills onto one replica (measured 5.7s).
- Model at `/workspace/models/GLM-5.2-MXFP4-FP8`; kill helper at `/workspace/kill_vllm.sh`.
- Replicas launch with `setsid` so they survive the launching shell exiting.

## Memory & max input (per TP4 instance, MI355X 288 GiB, util 0.85)
- Weights ~96 GiB/GPU, KV pool ~138.6 GiB = **3.1M tokens** capacity, activation ~10 GiB.
- KV ≈ 47.7 KB/token (MLA latent + sparse indexer; replicated across TP → per-GPU == per-instance).
- **Max prefill input is bounded by `--max-model-len`, not memory.** The model supports
  **1M** natively (`max_position_embeddings=1048576`, no rope scaling). A 1M-token request
  uses ~46.6 GiB of the 138.6 GiB KV pool (~3 concurrent 1M seqs/instance). To serve 1M,
  set `--max-model-len 1048576` in the replica script (default is 103424).

## Files
- `start_2tp4_lb.sh` / `start_2tp4_pure_lb.sh` — orchestrators (self-locating; start/stop).
- `vllm_glm_fp4_prefill_tp4ep4.sh` / `vllm_glm_fp4_prefill_tp4_pure.sh` — single-replica servers (honor `PORT`, `CUDA_VISIBLE_DEVICES`).
- `nginx_lb.conf` — LB config (listen :8000, upstream :8001/:8002, least_conn, SSE-friendly).
