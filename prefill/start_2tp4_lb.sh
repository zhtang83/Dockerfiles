#!/usr/bin/env bash
# Bring up 2x TP4+EP4 GLM-5.2 replicas + nginx least_conn LB.
#
#   Replica A : GPUs 0-3  -> 127.0.0.1:8001
#   Replica B : GPUs 4-7  -> 127.0.0.1:8002
#   LB (nginx): least_conn -> 127.0.0.1:8000   <- point clients here (OpenAI-compatible)
#
# Usage:
#   bash start_2tp4_lb.sh          # start everything, wait until ready
#   bash start_2tp4_lb.sh stop     # stop LB + both replicas
#
# Logs: /workspace/glm/srv_tp4ep4_a.log , srv_tp4ep4_b.log , /tmp/nginx_lb.err.log

set -u
GLM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPLICA_SH="${GLM_DIR}/vllm_glm_fp4_prefill_tp4ep4.sh"
NGINX_CONF="${GLM_DIR}/nginx_lb.conf"
LOG_A="${GLM_DIR}/srv_tp4ep4_a.log"
LOG_B="${GLM_DIR}/srv_tp4ep4_b.log"
LB_PORT=8000
HEALTH_TIMEOUT=600   # seconds to wait per replica

stop_all() {
    echo "[stop] quitting nginx..."
    nginx -c "${NGINX_CONF}" -s quit 2>/dev/null || true
    sleep 1
    echo "[stop] killing vLLM replicas..."
    bash /workspace/kill_vllm.sh 2>&1 | tail -1
    echo "[stop] done."
}

if [[ "${1:-}" == "stop" ]]; then
    stop_all
    exit 0
fi

# ---- preflight ----
for f in "${REPLICA_SH}" "${NGINX_CONF}"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done
command -v nginx >/dev/null 2>&1 || { echo "ERROR: nginx not installed (apt-get install -y nginx)" >&2; exit 1; }
if ps -eo comm | grep -q '^VLLM::'; then
    echo "ERROR: VLLM processes already running. Run 'bash start_2tp4_lb.sh stop' first." >&2
    exit 1
fi
mkdir -p /tmp/nginx_lb_body /tmp/nginx_lb_proxy

wait_health() {  # $1=port $2=label
    local port="$1" label="$2" i
    for ((i=0; i<HEALTH_TIMEOUT/5; i++)); do
        if curl -s -m3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            echo "[ready] ${label} healthy on :${port} (~$((i*5))s)"; return 0
        fi
        sleep 5
    done
    echo "ERROR: ${label} (:${port}) did not become healthy in ${HEALTH_TIMEOUT}s" >&2
    return 1
}

# ---- start replicas ----
echo "[start] replica A: GPUs 0-3 -> :8001"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=8001 nohup bash "${REPLICA_SH}" > "${LOG_A}" 2>&1 &
echo "[start] replica B: GPUs 4-7 -> :8002"
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=8002 nohup bash "${REPLICA_SH}" > "${LOG_B}" 2>&1 &

wait_health 8001 "replica A" || { echo "see ${LOG_A}"; exit 1; }
wait_health 8002 "replica B" || { echo "see ${LOG_B}"; exit 1; }

# ---- start LB ----
echo "[start] nginx LB (least_conn) -> :${LB_PORT}"
nginx -c "${NGINX_CONF}" -t >/dev/null 2>&1 || { echo "ERROR: nginx config test failed"; nginx -c "${NGINX_CONF}" -t; exit 1; }
nginx -c "${NGINX_CONF}"
sleep 1
if curl -s -m5 "http://127.0.0.1:${LB_PORT}/health" >/dev/null 2>&1; then
    echo "[ready] LB healthy on :${LB_PORT}"
else
    echo "ERROR: LB not responding on :${LB_PORT} (see /tmp/nginx_lb.err.log)" >&2
    exit 1
fi

echo
echo "==================================================================="
echo " READY. Point clients at:  http://127.0.0.1:${LB_PORT}"
echo "   e.g.  vllm bench serve --base-url http://127.0.0.1:${LB_PORT} ..."
echo " Stop everything with:     bash ${GLM_DIR}/start_2tp4_lb.sh stop"
echo "==================================================================="
