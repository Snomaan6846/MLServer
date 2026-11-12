#!/usr/bin/env bash
# =============================================================================
# bootstrap-cluster.sh — fresh-cluster setup from the checked-in assets
#
# Turns an empty OpenShift project into a working benchmarking environment:
# namespace, PVCs, RBAC, client image + pod, models on the model PVC, wire
# payloads, parity references and tooling on the client pod. Ends with both
# models ready and prints the next steps.
#
# Prerequisites: oc logged in (or CLUSTER_USER/CLUSTER_PASS/CLUSTER_URL in
# ~/.bashrc for auto re-login), podman for nothing (image builds happen in
# OpenShift), and this repository checked out.
#
# Usage: bash benchmarking/campaign/bootstrap-cluster.sh
# =============================================================================

set -euo pipefail

CTX="${CTX:-default/api-syedali-cluster-pool-q7c2q-gcp-rh-ods-com:6443/htpasswd-cluster-admin-user}"
NS="${NS:-mlserver-perf}"
RESOURCES="$(cd "$(dirname "$0")/../.." && pwd)"
ASSETS="${RESOURCES}/assets"
MANIFESTS="${RESOURCES}/benchmarking/cluster/manifests"
TOOLS="${RESOURCES}/benchmarking"

log() { echo "[bootstrap] $*"; }
die() { echo "[bootstrap] FAILED: $*" >&2; exit 1; }

relogin_cluster() {
    bash -c 'source ~/.bashrc && oc login -u "$CLUSTER_USER" -p "$CLUSTER_PASS" "$CLUSTER_URL" --insecure-skip-tls-verify=true' > /dev/null 2>&1
}

oc_ok() { oc --context "${CTX}" whoami > /dev/null 2>&1; }

ensure_login() {
    if ! oc_ok; then
        log "session expired, re-logging in..."
        relogin_cluster || die "cannot log in to the cluster (check ~/.bashrc credentials)"
    fi
    oc_ok || die "cluster unreachable"
}

# --- 1. namespace ------------------------------------------------------------
ensure_login
if ! oc --context "${CTX}" get namespace "${NS}" > /dev/null 2>&1; then
    oc --context "${CTX}" create namespace "${NS}" > /dev/null
    log "created namespace ${NS}"
else
    log "namespace ${NS} exists"
fi
OC="oc --context ${CTX} -n ${NS}"

# --- 2. storage, RBAC, client build ------------------------------------------
log "applying PVC, RBAC and client-build manifests..."
${OC} apply -f "${MANIFESTS}/pvc-rfdetr-models.yaml" > /dev/null
${OC} apply -f "${MANIFESTS}/sweep-rbac.yaml" > /dev/null
${OC} apply -f "${MANIFESTS}/client-build.yaml" > /dev/null

${OC} wait pvc/rfdetr-models --for=bound --timeout=300s > /dev/null \
    || die "rfdetr-models PVC did not bind"
log "model PVC bound"

# --- 3. client image + pod ----------------------------------------------------
# The BuildConfig uses a binary source: assemble the context (Dockerfile +
# tool files) and start the build.
CTXDIR="$(mktemp -d)"
trap 'rm -rf "${CTXDIR}"' EXIT
cp "${RESOURCES}/benchmarking/overlay/Dockerfile.client" "${CTXDIR}/Dockerfile"
mkdir -p "${CTXDIR}/tools"
cp "${TOOLS}/cluster/hook.py" "${TOOLS}/cluster/triton_hook.py" \
   "${TOOLS}/study_tools/sweep.py" "${TOOLS}/study_tools/client.py" \
   "${TOOLS}/study_tools/analyze.py" "${TOOLS}/study_tools/payloads.py" \
   "${CTXDIR}/tools/"
log "building client image (binary build)..."
${OC} start-build mlserver-client --from-dir="${CTXDIR}" --wait > /dev/null \
    || die "client image build failed"
${OC} apply -f "${MANIFESTS}/client-deploy.yaml" > /dev/null
${OC} rollout status deployment/mlserver-client --timeout=300s > /dev/null \
    || die "client deployment did not become ready"
CLIENT_POD=$(${OC} get pods -l app=mlserver-client \
    -o jsonpath='{.items[0].metadata.name}')
log "client pod ready: ${CLIENT_POD}"

# --- 4. models onto the PVC (via the model-loader pod) ------------------------
# Models are SHA-pinned in assets/MANIFEST.json. They are not committed
# (GitHub LFS is unavailable on public forks); on a fresh checkout the
# bootstrap regenerates them from the pinned export environment.
if [[ ! -s "${ASSETS}/models/rfdetr-small/model.onnx" ]] \
    || [[ ! -s "${ASSETS}/models/rfdetr-small-batched/model.onnx" ]]; then
    log "models not present — regenerating from the pinned export environment..."
    log "(first run downloads the public RF-DETR weights and exports; ~10-15 min)"
    (cd "${RESOURCES}/model-preparation/export-env" \
        && poetry run python "${RESOURCES}/benchmarking/campaign/export-models.py") \
        || die "model export failed"
fi
test -s "${ASSETS}/models/rfdetr-small/model.onnx" \
    || die "assets/models/rfdetr-small/model.onnx missing"
test -s "${ASSETS}/models/rfdetr-small-batched/model.onnx" \
    || die "assets/models/rfdetr-small-batched/model.onnx missing"

log "starting model-loader pod..."
${OC} apply -f "${MANIFESTS}/model-loader.yaml" > /dev/null
${OC} wait pod/model-loader --for=condition=Ready --timeout=300s > /dev/null \
    || die "model-loader pod not ready"

log "uploading models and PVC state..."
${OC} exec model-loader -- mkdir -p \
    /mnt/models/rfdetr-small /mnt/models/rfdetr-small-batched \
    /mnt/models/triton-repo/rfdetr-small
cat "${ASSETS}/cluster-state/pvc-settings.json" | \
    ${OC} exec -i model-loader -- sh -c 'cat > /mnt/models/settings.json'
cat "${ASSETS}/models/rfdetr-small/model.onnx" | \
    ${OC} exec -i model-loader -- sh -c 'cat > /mnt/models/rfdetr-small/model.onnx'
cat "${ASSETS}/models/rfdetr-small/model-settings.json" | \
    ${OC} exec -i model-loader -- sh -c 'cat > /mnt/models/rfdetr-small/model-settings.json'
cat "${ASSETS}/models/rfdetr-small-batched/model.onnx" | \
    ${OC} exec -i model-loader -- sh -c 'cat > /mnt/models/rfdetr-small-batched/model.onnx'
cat "${ASSETS}/models/rfdetr-small-batched/model-settings.json" | \
    ${OC} exec -i model-loader -- sh -c 'cat > /mnt/models/rfdetr-small-batched/model-settings.json'
cat "${ASSETS}/cluster-state/triton-repo/rfdetr-small/config.pbtxt" | \
    ${OC} exec -i model-loader -- sh -c 'cat > /mnt/models/triton-repo/rfdetr-small/config.pbtxt'
# Triton serves the same model bytes
cat "${ASSETS}/models/rfdetr-small/model.onnx" | \
    ${OC} exec -i model-loader -- sh -c 'mkdir -p /mnt/models/triton-repo/rfdetr-small/1 && cat > /mnt/models/triton-repo/rfdetr-small/1/model.onnx'

log "verifying model SHAs against assets/MANIFEST.json..."
verify_sha() {  # local_file remote_path
    local expect remote
    expect=$(python3 -c "
import json
print(json.load(open('${ASSETS}/MANIFEST.json'))['${1#assets/}'])
")
    remote=$(${OC} exec model-loader -- sha256sum "${2}" | cut -d' ' -f1)
    [[ "${expect}" == "${remote}" ]] || die "SHA mismatch for ${2}: ${remote} != ${expect}"
}
verify_sha "${ASSETS}/models/rfdetr-small/model.onnx" /mnt/models/rfdetr-small/model.onnx
verify_sha "${ASSETS}/models/rfdetr-small-batched/model.onnx" /mnt/models/rfdetr-small-batched/model.onnx
log "model SHAs verified"

${OC} delete pod model-loader --grace-period=0 --force > /dev/null 2>&1 || true
log "model-loader removed"

# --- 5. serving deployment ----------------------------------------------------
log "deploying the serving deployment..."
${OC} apply -f "${MANIFESTS}/mlserver-baseline.yaml" > /dev/null
${OC} rollout status deployment/mlserver-rfdetr --timeout=600s > /dev/null \
    || die "serving deployment did not become ready"

for model in rfdetr-small rfdetr-small-batched; do
    ${OC} exec "${CLIENT_POD}" -- python3 -c "
import sys, time, urllib.request
deadline = time.time() + 300
while time.time() < deadline:
    try:
        with urllib.request.urlopen(
            'http://mlserver-rfdetr.${NS}.svc:8080/v2/models/${model}/ready',
            timeout=5,
        ) as r:
            if r.status == 200:
                print('${model}: ready')
                sys.exit(0)
    except Exception:
        pass
    time.sleep(2)
print('${model}: NOT ready in time')
sys.exit(1)
" || die "model ${model} not ready"
done

# --- 6. client-pod state: payloads, references, tooling -----------------------
log "uploading payloads, references and tooling to the client pod..."
${OC} exec "${CLIENT_POD}" -- mkdir -p /store/payloads /store/references \
    /store/tools /store/plans /store/sessions
for f in "${ASSETS}"/payloads/*; do
    ${OC} cp "${f}" "${CLIENT_POD}:/store/payloads/$(basename "${f}")"
done
for f in "${ASSETS}"/references/*; do
    ${OC} cp "${f}" "${CLIENT_POD}:/store/references/$(basename "${f}")"
done
${OC} cp "${ASSETS}/cluster-state/images.json" "${CLIENT_POD}:/store/images.json"
for t in hook.py triton_hook.py; do
    ${OC} cp "${TOOLS}/cluster/${t}" "${CLIENT_POD}:/store/tools/${t}"
done
for t in sweep.py client.py analyze.py payloads.py; do
    ${OC} cp "${TOOLS}/study_tools/${t}" "${CLIENT_POD}:/store/tools/${t}"
done
log "client-pod state uploaded"

# --- done ----------------------------------------------------------------------
log ""
log "BOOTSTRAP COMPLETE"
log "  namespace:      ${NS}"
log "  client pod:     ${CLIENT_POD} (/store: payloads, references, tools)"
log "  serving:        mlserver-rfdetr (both models ready)"
log "  model PVC:      rfdetr-models (settings, models, triton-repo)"
log ""
log "Next steps:"
log "  1. Build the image you want to benchmark (see benchmarking/overlay/)"
log "     and push it to a registry the cluster can pull."
log "  2. Write variants + plans (benchmarking/campaign/write-variants.py,"
log "     write-plans.py) and upload variants.json to the client pod."
log "  3. Run the campaign: benchmarking/campaign/run-final-campaign.sh"
log "     — or individual sweeps; see BENCHMARKING_GUIDE.md sections 6-7."
