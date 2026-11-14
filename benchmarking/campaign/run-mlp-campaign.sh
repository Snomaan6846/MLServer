#!/usr/bin/env bash
set -euo pipefail

CTX="default/api-syedali-cluster-pool-q7c2q-gcp-rh-ods-com:6443/htpasswd-cluster-admin-user"
NS="mlserver-perf"
RESOURCES="/home/syedali/Workspace/mlserver-performance/MLServer-resources"
CAMP="${RESOURCES}/benchmark-results/mlp-campaign"
ARTIFACTS="/tmp/opencode/campaign-b"
CLIENT_POD="mlserver-client-6c7fc47bbf-7g2t2"

log() { echo "[mlp] $(date -u +%FT%TZ) $*" | tee -a "${CAMP}/logs/session.log"; }
die() { echo "[mlp] FAILED: $*" | tee -a "${CAMP}/logs/session.log" >&2; exit 1; }

relogin() { bash -c 'source ~/.bashrc && oc login -u "$CLUSTER_USER" -p "$CLUSTER_PASS" "$CLUSTER_URL" --insecure-skip-tls-verify=true' > /dev/null 2>&1; }
ensure_login() {
    local n=0
    while ! oc --context "${CTX}" whoami > /dev/null 2>&1; do
        n=$((n+1)); [[ $n -gt 5 ]] && die "cluster unreachable"
        log "  > re-logging in"
        relogin || sleep 10
    done
}

retry() {
    local desc="$1"; shift
    local n=0
    until "$@" > /dev/null 2>&1; do
        n=$((n+1)); [[ $n -gt 5 ]] && { log "  > ${desc}: FAILED"; return 1; }
        log "  > ${desc}: attempt ${n} failed, retrying..."
        ensure_login; sleep $((5 * n))
    done
}

run_session() {
    local name="$1"
    local sdir="${CAMP}/sessions/${name}"
    log "=== SESSION ${name} ==="
    ensure_login
    oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- rm -rf "/store/sessions/${name}" 2>/dev/null || true
    rm -rf "${sdir}"; mkdir -p "${sdir}"

    log "  > uploading plan"
    retry "upload plan" oc --context "${CTX}" -n "${NS}" cp "${CAMP}/plan-${name}.json" "${CLIENT_POD}:/store/plans/${name}.json"

    log "  > starting sampler"
    python3 "${RESOURCES}/benchmarking/cluster/sampler.py" \
        --output "${CAMP}/samples/${name}.jsonl" --interval 3 --duration 14400 &
    local SPID=$!

    log "  > running sweep"
    local attempt=0
    while ! oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- \
        python3 /store/tools/sweep.py --plan "/store/plans/${name}.json" \
        --output "/store/sessions/${name}" > "${CAMP}/logs/${name}-sweep.log" 2>&1; do
        attempt=$((attempt+1))
        if [[ $attempt -gt 3 ]]; then
            log "  > WARNING: sweep ${name} failed after retries — collecting what exists"
            break
        fi
        log "  > sweep attempt ${attempt} failed — retrying"
        ensure_login; sleep 30
        oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- rm -rf "/store/sessions/${name}" 2>/dev/null || true
    done
    kill "${SPID}" 2>/dev/null || true
    wait "${SPID}" 2>/dev/null || true

    log "  > collecting"
    local n=0
    until oc --context "${CTX}" -n "${NS}" cp "${CLIENT_POD}:/store/sessions/${name}" "${sdir}" > /dev/null 2>&1; do
        n=$((n+1)); [[ $n -gt 3 ]] && { log "  > WARNING: collect ${name} failed"; return 0; }
        log "  > collect retry ${n}"; sleep 15
    done

    log "  > summarizing"
    python3 "${RESOURCES}/benchmarking/cluster/summarize.py" \
        --session "${sdir}" --sampler "${CAMP}/samples/${name}.jsonl" >> "${CAMP}/logs/session.log" 2>&1 || true
    local V
    V=$(python3 -c "
import json
d = json.load(open('${sdir}/SUMMARY.json'))
invalid = [r for r in d['results'] if not r['valid']]
print(len(invalid))
" 2>/dev/null) || V="unknown"
    if [[ "${V}" != "0" ]]; then
        log "  > WARNING: ${V} invalid trials in ${name}"
    else
        log "  > all trials valid"
    fi

    python3 -c "
import json
from collections import defaultdict
import numpy as np
d = json.load(open('${sdir}/SUMMARY.json'))
groups = defaultdict(list)
for r in d['results']:
    if r['valid']:
        groups[(r['variant'], r['concurrency'])].append(r['rps'])
for (v, c), rates in sorted(groups.items()):
    print(f'  {v:34s} C{c:3d}  {np.median(rates):9.2f} RPS')
" 2>/dev/null | tee -a "${CAMP}/logs/session.log" || true

    log "  > ${name} COMPLETE"
}

server_pod() {
    oc --context "${CTX}" -n "${NS}" get pods -l app=mlserver-rfdetr-cpu \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true
}

ensure_login
mkdir -p "${CAMP}/logs" "${CAMP}/sessions" "${CAMP}/samples"

# ------------------------------------------------------------------
# Step 0: upload tooling, payloads, variants, plans
# ------------------------------------------------------------------
log "STEP 0: upload tooling + payloads"
retry "upload hook" oc --context "${CTX}" -n "${NS}" cp \
    "${RESOURCES}/benchmarking/cluster/hook.py" "${CLIENT_POD}:/store/tools/hook.py"
for f in "${ARTIFACTS}"/*.bin "${ARTIFACTS}"/binary-frames.json; do
    retry "upload $(basename ${f})" oc --context "${CTX}" -n "${NS}" cp \
        "${f}" "${CLIENT_POD}:/store/payloads/$(basename ${f})"
done
retry "upload variants" oc --context "${CTX}" -n "${NS}" cp \
    "${CAMP}/variants.json" "${CLIENT_POD}:/store/variants.json"
for plan in "${CAMP}"/plan-*.json; do
    retry "upload $(basename ${plan})" oc --context "${CTX}" -n "${NS}" cp \
        "${plan}" "${CLIENT_POD}:/store/plans/$(basename ${plan} .json).json"
done

# Guard: kill any stale sweep left by a previously killed run.
oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- python3 -c "
import os, signal
killed = 0
for pid in os.listdir('/proc'):
    if not pid.isdigit():
        continue
    try:
        cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\x00', b' ').decode('utf-8', 'replace')
        if '/store/tools/sweep.py' in cmd:
            os.kill(int(pid), signal.SIGKILL)
            killed += 1
    except Exception:
        pass
print(f'stale sweeps killed: {killed}')
" 2>/dev/null || true
log "  > uploaded (stale sweeps cleared)"

# ------------------------------------------------------------------
# Step 1: single-model deployment — stash the RF-DETR models, upload
# the MLP (phase-1 fidelity: one model, like the original study)
# ------------------------------------------------------------------
log "STEP 1: prepare single-model deployment"
oc --context "${CTX}" -n "${NS}" scale deploy/mlserver-rfdetr --replicas=0 > /dev/null 2>&1 || true
oc --context "${CTX}" -n "${NS}" scale deploy/mlserver-rfdetr-cpu --replicas=1 > /dev/null
sleep 10
SRV=$(server_pod)
[[ -n "${SRV}" ]] || die "no CPU server pod for PVC preparation"
oc --context "${CTX}" -n "${NS}" exec "${SRV}" -- mv /mnt/models/rfdetr-small /mnt/models/.rfdetr-small-stash 2>/dev/null || true
oc --context "${CTX}" -n "${NS}" exec "${SRV}" -- mv /mnt/models/rfdetr-small-batched /mnt/models/.rfdetr-small-batched-stash 2>/dev/null || true
oc --context "${CTX}" -n "${NS}" exec "${SRV}" -- mkdir -p /mnt/models/mlp-small
# The server container has no tar (oc cp needs it) — stream via stdin.
oc --context "${CTX}" -n "${NS}" exec -i "${SRV}" -- \
    python3 -c "import sys; open('/mnt/models/mlp-small/model.onnx','wb').write(sys.stdin.buffer.read())" \
    < "${ARTIFACTS}/model.onnx" || die "model.onnx upload failed"
oc --context "${CTX}" -n "${NS}" exec -i "${SRV}" -- \
    python3 -c "import sys; open('/mnt/models/mlp-small/model-settings.json','wb').write(sys.stdin.buffer.read())" \
    < "${ARTIFACTS}/model-settings.json" || die "model-settings upload failed"
oc --context "${CTX}" -n "${NS}" exec "${SRV}" -- \
    python3 -c "
import os
size = os.path.getsize('/mnt/models/mlp-small/model.onnx')
assert size == 2985522, f'unexpected model size: {size}'
print(f'model verified on PVC: {size} bytes')
" | tee -a "${CAMP}/logs/session.log"
log "  > mlp-small deployed (single-model conditions)"

# ------------------------------------------------------------------
# Step 2: capture the MLP references (one-time, from this image)
# ------------------------------------------------------------------
log "STEP 2: capture mlp-small references"
oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- rm -rf /store/capture-mlp 2>/dev/null || true
oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- mkdir -p /store/capture-mlp
oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- env BENCHMARK_TRIAL_DIR=/store/capture-mlp \
    python3 /store/tools/hook.py mlp-bin-worker --capture-refs \
    > "${CAMP}/logs/capture-mlp.log" 2>&1 || die "mlp reference capture failed"
tail -2 "${CAMP}/logs/capture-mlp.log" | tee -a "${CAMP}/logs/session.log"

# ------------------------------------------------------------------
# Step 3: run the campaign sessions
# ------------------------------------------------------------------
run_session mlp-B2-core
run_session mlp-B3-wsweep
run_session mlp-B4-csweep
run_session mlp-B5-transport
run_session mlp-B6-threads

# ------------------------------------------------------------------
# Step 4: restore the original PVC state and park the deployments
# ------------------------------------------------------------------
log "STEP 4: restore PVC state"
SRV=$(server_pod)
if [[ -n "${SRV}" ]]; then
    oc --context "${CTX}" -n "${NS}" exec "${SRV}" -- rm -rf /mnt/models/mlp-small 2>/dev/null || true
    oc --context "${CTX}" -n "${NS}" exec "${SRV}" -- mv /mnt/models/.rfdetr-small-stash /mnt/models/rfdetr-small 2>/dev/null || true
    oc --context "${CTX}" -n "${NS}" exec "${SRV}" -- mv /mnt/models/.rfdetr-small-batched-stash /mnt/models/rfdetr-small-batched 2>/dev/null || true
fi
oc --context "${CTX}" -n "${NS}" scale deploy/mlserver-rfdetr-cpu --replicas=0 > /dev/null 2>&1 || true
oc --context "${CTX}" -n "${NS}" patch deploy mlserver-rfdetr-cpu --type=json -p '[{"op":"replace","path":"/spec/template/spec/containers/0/env","value":[{"name":"MLSERVER_MODEL_ONNX_PROVIDERS","value":"[\"CPUExecutionProvider\"]"},{"name":"MLSERVER_PARALLEL_WORKERS","value":"1"},{"name":"MLSERVER_REST_BINARY_TENSOR","value":"true"},{"name":"MLSERVER_PARALLEL_INFERENCE_TRANSPORT","value":"zmq"},{"name":"MLSERVER_MODEL_ONNX_SESSION_OPTIONS","value":"{\"intra_op_num_threads\": 2, \"inter_op_num_threads\": 1}"},{"name":"MLSERVER_MODEL_ONNX_SESSION_CONFIG_ENTRIES","value":"{\"session.intra_op.allow_spinning\": \"0\", \"session.inter_op.allow_spinning\": \"0\"}"}]}]' > /dev/null 2>&1 || true
oc --context "${CTX}" -n "${NS}" scale deploy/mlserver-rfdetr --replicas=1 > /dev/null 2>&1 || true
oc --context "${CTX}" -n "${NS}" rollout status deploy/mlserver-rfdetr --timeout=300s > /dev/null 2>&1 || true

log ""
log "MLP CAMPAIGN B COMPLETE"
log "  results: ${CAMP}"
