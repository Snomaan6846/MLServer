#!/usr/bin/env bash
set -euo pipefail

CTX="default/api-syedali-cluster-pool-q7c2q-gcp-rh-ods-com:6443/htpasswd-cluster-admin-user"
NS="mlserver-perf"
RESOURCES="/home/syedali/Workspace/mlserver-performance/MLServer-resources"
CAMP="${RESOURCES}/benchmark-results/worker-binary-campaign"
CLIENT_POD="mlserver-client-6c7fc47bbf-7g2t2"

log() { echo "[wb] $(date -u +%FT%TZ) $*" | tee -a "${CAMP}/logs/session.log"; }
die() { echo "[wb] FAILED: $*" | tee -a "${CAMP}/logs/session.log" >&2; exit 1; }

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
    local DUR=4000
    [[ "${name}" == "wb-A6-soak" ]] && DUR=4500
    python3 "${RESOURCES}/benchmarking/cluster/sampler.py" \
        --output "${CAMP}/samples/${name}.jsonl" --interval 3 --duration "${DUR}" &
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
    print(f'  {v:28s} C{c:2d}  {np.median(rates):7.2f} RPS')
" 2>/dev/null | tee -a "${CAMP}/logs/session.log" || true

    log "  > ${name} COMPLETE"
}

ensure_login
mkdir -p "${CAMP}/logs" "${CAMP}/sessions" "${CAMP}/samples"

# Park the sibling deployment first: its graceful termination can delay
# the first GPU rollout past the hook's 420s budget (two 28-CPU
# Guaranteed pods cannot co-schedule on this node).
oc --context "${CTX}" -n "${NS}" scale deploy/mlserver-rfdetr-cpu --replicas=0 > /dev/null 2>&1 || true
oc --context "${CTX}" -n "${NS}" rollout status deploy/mlserver-rfdetr-cpu --timeout=120s > /dev/null 2>&1 || true
log "sibling CPU deployment parked"

# Self-heal: a previously killed run may have left the batched model
# stashed — restore it so the Step 1 batched capture can find it.
oc --context "${CTX}" -n "${NS}" rollout restart deploy/mlserver-rfdetr > /dev/null 2>&1 || true
oc --context "${CTX}" -n "${NS}" rollout status deploy/mlserver-rfdetr --timeout=300s > /dev/null 2>&1 || true
SRV=$(oc --context "${CTX}" -n "${NS}" get pods -l app=mlserver-rfdetr -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [[ -n "${SRV}" ]]; then
    oc --context "${CTX}" -n "${NS}" exec "${SRV}" -- \
        mv /mnt/models/.rfdetr-small-batched-stash /mnt/models/rfdetr-small-batched 2>/dev/null \
        && log "batched model restored from stash (self-heal)" || true
    oc --context "${CTX}" -n "${NS}" exec "${SRV}" -- \
        mv /mnt/models/.rfdetr-small-stash /mnt/models/rfdetr-small 2>/dev/null \
        && log "default model restored from stash (self-heal)" || true
fi

# ------------------------------------------------------------------
# Step 0: upload tooling + variants + plans
# ------------------------------------------------------------------
log "STEP 0: upload tooling"
retry "upload hook" oc --context "${CTX}" -n "${NS}" cp \
    "${RESOURCES}/benchmarking/cluster/hook.py" "${CLIENT_POD}:/store/tools/hook.py"
retry "upload binary-frames sidecar" oc --context "${CTX}" -n "${NS}" cp \
    "/tmp/opencode/campaign-b/binary-frames.json" "${CLIENT_POD}:/store/payloads/binary-frames.json"
retry "upload variants" oc --context "${CTX}" -n "${NS}" cp \
    "${CAMP}/variants.json" "${CLIENT_POD}:/store/variants.json"
for plan in "${CAMP}"/plan-*.json; do
    retry "upload $(basename ${plan})" oc --context "${CTX}" -n "${NS}" cp \
        "${plan}" "${CLIENT_POD}:/store/plans/$(basename ${plan} .json).json"
done

# Guard: kill any stale sweep left by a previously killed run. A zombie
# sweep fights the new one for the deployment and corrupts the session
# (observed: interleaved case numbering and contaminated SUMMARY rows).
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
# Step 1: reference refresh with cross-image stability proof
#   - back up the frozen references
#   - capture fresh references on the new image (both models)
#   - byte-compare: every pre-existing reference must be IDENTICAL
#     (proves our code changes did not alter any output); the binary
#     references are new and become the campaign's parity gates
# ------------------------------------------------------------------
log "STEP 1: reference refresh + cross-image stability proof"
# Idempotent: a previous partial run may have overwritten references —
# always start from the frozen backup if one exists.
if oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- test -d /store/references-frozen-backup 2>/dev/null; then
    log "  > restoring references from the existing frozen backup"
    oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- rm -rf /store/references 2>/dev/null || true
    oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- cp -r /store/references-frozen-backup /store/references
else
    oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- cp -r /store/references /store/references-frozen-backup
fi

log "  > capturing references on the new image (default model)"
oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- rm -rf /store/capture-default 2>/dev/null || true
oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- mkdir -p /store/capture-default
oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- env BENCHMARK_TRIAL_DIR=/store/capture-default \
    python3 /store/tools/hook.py wb-binary-frontend --capture-refs \
    > "${CAMP}/logs/capture-default.log" 2>&1 || die "default-model reference capture failed"
tail -2 "${CAMP}/logs/capture-default.log" | tee -a "${CAMP}/logs/session.log"

log "  > capturing references on the new image (batched model)"
oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- rm -rf /store/capture-batched 2>/dev/null || true
oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- mkdir -p /store/capture-batched
oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- env BENCHMARK_TRIAL_DIR=/store/capture-batched \
    python3 /store/tools/hook.py wb-batched-frontend --capture-refs \
    > "${CAMP}/logs/capture-batched.log" 2>&1 || die "batched-model reference capture failed"
tail -2 "${CAMP}/logs/capture-batched.log" | tee -a "${CAMP}/logs/session.log"

log "  > comparing frozen vs fresh references (must be byte-identical)"
oc --context "${CTX}" -n "${NS}" exec -i "${CLIENT_POD}" -- python3 - << 'COMPARE' | tee -a "${CAMP}/logs/session.log"
import hashlib
from pathlib import Path

frozen = Path("/store/references-frozen-backup")
fresh = Path("/store/references")
identical, new_files, changed = [], [], []
for ref in sorted(fresh.iterdir()):
    if not ref.is_file():
        continue
    old = frozen / ref.name
    if not old.exists():
        new_files.append(ref.name)
    elif old.read_bytes() == ref.read_bytes():
        identical.append(ref.name)
    else:
        changed.append(ref.name)
print(f"  references: {len(identical)} identical, {len(new_files)} new, {len(changed)} CHANGED")
for name in new_files:
    print(f"    new: {name}")
for name in changed:
    print(f"    CHANGED: {name} (frozen {hashlib.sha256((frozen/name).read_bytes()).hexdigest()[:16]}, fresh {hashlib.sha256((fresh/name).read_bytes()).hexdigest()[:16]})")
if changed:
    raise SystemExit("OUTPUT REGRESSION: references differ across images")
print("  cross-image output stability: PROVEN")
COMPARE
[ "${PIPESTATUS[0]}" -eq 0 ] || die "reference comparison failed — output regression"

# ------------------------------------------------------------------
# Step 2: run the campaign sessions (single-model conditions for the
# unbatched sessions — the batched model is stashed, matching the final
# campaign's discipline; restored only for A4)
# ------------------------------------------------------------------
stash_batched() {
    local SRV
    SRV=$(oc --context "${CTX}" -n "${NS}" get pods -l app=mlserver-rfdetr \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [[ -n "${SRV}" ]]; then
        oc --context "${CTX}" -n "${NS}" exec "${SRV}" -- \
            mv /mnt/models/rfdetr-small-batched /mnt/models/.rfdetr-small-batched-stash 2>/dev/null || true
    fi
}
restore_batched() {
    local SRV
    SRV=$(oc --context "${CTX}" -n "${NS}" get pods -l app=mlserver-rfdetr \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [[ -n "${SRV}" ]]; then
        oc --context "${CTX}" -n "${NS}" exec "${SRV}" -- \
            mv /mnt/models/.rfdetr-small-batched-stash /mnt/models/rfdetr-small-batched 2>/dev/null || true
    fi
}

stash_batched
log "  > batched model stashed (single-model conditions)"
run_session wb-A1-core
run_session wb-A2-jsonworker
run_session wb-A3-orthogonality
run_session wb-A5-graph
restore_batched
log "  > batched model restored (A4)"
run_session wb-A4-batched
stash_batched
log "  > batched model stashed (soak under single-model conditions)"
run_session wb-A6-soak
restore_batched
log "  > batched model restored (final state)"

# ------------------------------------------------------------------
# Step 3: hygiene — reset the GPU deployment to the frozen baseline
# ------------------------------------------------------------------
log "STEP 3: reset GPU deployment to frozen baseline"
oc --context "${CTX}" -n "${NS}" scale deploy/mlserver-rfdetr-cpu --replicas=0 > /dev/null 2>&1 || true
oc --context "${CTX}" -n "${NS}" patch deploy mlserver-rfdetr --type=json -p '[{"op":"replace","path":"/spec/template/spec/containers/0/env","value":[{"name":"MLSERVER_MODEL_ONNX_PROVIDERS","value":"[\"CUDAExecutionProvider\",\"CPUExecutionProvider\"]"},{"name":"MLSERVER_PARALLEL_WORKERS","value":"1"},{"name":"MLSERVER_REST_BINARY_TENSOR","value":"true"},{"name":"MLSERVER_PARALLEL_INFERENCE_TRANSPORT","value":"zmq"},{"name":"MLSERVER_MODEL_ONNX_SESSION_OPTIONS","value":"{\"intra_op_num_threads\": 2, \"inter_op_num_threads\": 1}"},{"name":"MLSERVER_MODEL_ONNX_SESSION_CONFIG_ENTRIES","value":"{\"session.intra_op.allow_spinning\": \"0\", \"session.inter_op.allow_spinning\": \"0\"}"}]}]' > /dev/null
oc --context "${CTX}" -n "${NS}" rollout status deploy/mlserver-rfdetr --timeout=300s > /dev/null 2>&1 || true

log ""
log "WORKER-BINARY CAMPAIGN A COMPLETE"
log "  results: ${CAMP}"
