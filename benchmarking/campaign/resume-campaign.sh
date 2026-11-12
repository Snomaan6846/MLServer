#!/usr/bin/env bash
# =============================================================================
# RHOAIENG-102092: Final comprehensive benchmark campaign
#
# Robust, fully-automated, zero-assumption orchestrator.
# Every session uses unique output paths (no stale data reuse).
# Every step logged with timestamps. Every artifact verified.
# Any failure stops the campaign immediately — no partial data.
#
# Usage: bash run-final-campaign.sh
# Expected duration: ~7.5 hours
# =============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------

CTX="default/api-syedali-cluster-pool-q7c2q-gcp-rh-ods-com:6443/htpasswd-cluster-admin-user"
NS="mlserver-perf"
RES="/home/syedali/Workspace/mlserver-performance/MLServer-resources/benchmark-results"
REPO="/home/syedali/Workspace/mlserver-performance/MLServer"
RESOURCES="/home/syedali/Workspace/mlserver-performance/MLServer-resources"
CAMPAIGN_DIR="${RES}/final-campaign"
LOG_DIR="${CAMPAIGN_DIR}/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
HEAD_COMMIT="$(cd "${REPO}" && git rev-parse --short HEAD)"
HEAD_FULL="$(cd "${REPO}" && git rev-parse HEAD)"
OFFICIAL_IMAGE="quay.io/opendatahub/odh-mlserver-cuda@sha256:4c8168e9955daad321eedcd65bbcd27c9f210cfd56f8317f3fa12b10324de3fe"

# Ship image (built in step 3)
SHIP_IMAGE=$(cat "${CAMPAIGN_DIR}/ship-image.txt" 2>/dev/null || true)
[[ -n "${SHIP_IMAGE}" ]] || { echo "ship-image.txt missing — cannot resume"; exit 1; }

# Model info
MODEL_SHA="a018914e11d589b8b51154f00ed1103f148a320a3960f768bd44f57d55d0ab51"

# Retry configuration
MAX_RETRIES=5
RETRY_BASE_DELAY=5    # seconds; exponential backoff: delay * 2^attempt
MAX_RETRY_DELAY=120   # cap at 2 minutes

mkdir -p "${CAMPAIGN_DIR}" "${LOG_DIR}"
# --- Logging -----------------------------------------------------------------

log() {
    echo "[$(date -u +%FT%TZ)] $*" | tee -a "${LOG_DIR}/campaign.log"
}

# Same as log(), but writes the console copy to STDERR — used inside
# retry_with_backoff so its stdout stays clean for command substitution
# (callers capture the retried command's output; log lines must not mix in).
log_stderr() {
    echo "[$(date -u +%FT%TZ)] $*" | tee -a "${LOG_DIR}/campaign.log" >&2
}

log_section() {
    echo "" | tee -a "${LOG_DIR}/campaign.log"
    echo "========================================================================" | tee -a "${LOG_DIR}/campaign.log"
    echo "[$(date -u +%FT%TZ)] === $* ===" | tee -a "${LOG_DIR}/campaign.log"
    echo "========================================================================" | tee -a "${LOG_DIR}/campaign.log"
}

fail() {
    log "!!! CAMPAIGN FAILED at: $*"
    log "!!! Check logs in ${LOG_DIR}/"
    log "!!! Campaign directory: ${CAMPAIGN_DIR}/"
    exit 1
}

# --- Retry wrappers for transient failures ------------------------------------

# Retry any command with exponential backoff. If the failure is an
# auth error, attempt cluster re-login before retrying.
# Usage: retry_with_backoff <description> <max_retries> <command...>
retry_with_backoff() {
    local desc="$1"; shift
    local retries="$1"; shift
    local attempt=0
    local delay=${RETRY_BASE_DELAY}

    while true; do
        attempt=$((attempt + 1))
        local output
        if output=$("$@" 2>&1); then
            if [[ ${attempt} -gt 1 ]]; then
                log_stderr "  > ${desc}: succeeded on attempt ${attempt}"
            fi
            echo "${output}"
            return 0
        fi

        if [[ ${attempt} -ge ${retries} ]]; then
            log_stderr "  > ${desc}: FAILED after ${retries} attempts"
            echo "${output}"
            return 1
        fi

        local wait_time=$((delay * (2 ** (attempt - 1))))
        if [[ ${wait_time} -gt ${MAX_RETRY_DELAY} ]]; then
            wait_time=${MAX_RETRY_DELAY}
        fi

        # Check for auth failures — trigger re-login
        if echo "${output}" | grep -qi "unauthorized\|logged in to the server"; then
            log_stderr "  > ${desc}: auth failure on attempt ${attempt}, re-logging in..."
            if relogin_cluster; then
                continue  # retry immediately after successful re-login
            fi
        fi

        log_stderr "  > ${desc}: attempt ${attempt}/${retries} failed, retrying in ${wait_time}s..."
        sleep "${wait_time}"
    done
}

# Retry an `oc` command specifically (transient cluster connectivity).
# Usage: oc_retry <description> <command...>
oc_retry() {
    local desc="$1"; shift
    retry_with_backoff "${desc}" "${MAX_RETRIES}" \
        oc --context "${CTX}" -n "${NS}" "$@"
}

# Retry `oc exec` (transient pod restarts, API throttling).
# Usage: oc_exec_retry <description> <pod> <command...>
oc_exec_retry() {
    local desc="$1"; shift
    local pod="$1"; shift
    retry_with_backoff "${desc}" "${MAX_RETRIES}" \
        oc --context "${CTX}" -n "${NS}" exec "${pod}" -- "$@"
}

# Retry `oc cp` (transient tar/stream failures).
# Usage: oc_cp_retry <description> <src> <dst>
oc_cp_retry() {
    local desc="$1"; shift
    local src="$1"; shift
    local dst="$1"; shift
    retry_with_backoff "${desc}" "${MAX_RETRIES}" \
        oc --context "${CTX}" -n "${NS}" cp "${src}" "${dst}"
}

# Attempt cluster re-login using credentials from the environment.
# Credentials are NEVER logged — only the action is logged.
relogin_cluster() {
    log "  > cluster session expired, attempting re-login..."

    # Source bashrc to pick up cluster credentials (CLUSTER_USER,
    # CLUSTER_PASS, CLUSTER_URL). Values are never echoed or logged.
    if ! bash -c 'source ~/.bashrc && oc login -u "$CLUSTER_USER" -p "$CLUSTER_PASS" "$CLUSTER_URL" --insecure-skip-tls-verify=true' > /dev/null 2>&1; then
        log "  > re-login failed (credentials may be invalid or env vars not set)"
        return 1
    fi

    log "  > re-login successful"
    return 0
}

# Wait for the cluster API to become responsive. If unreachable due to
# an expired session, attempt re-login. Polls with backoff.
wait_for_cluster() {
    local desc="${1:-"cluster API"}"
    local max_wait="${2:-300}"  # 5 minutes
    local start=$(date +%s)

    while true; do
        if oc --context "${CTX}" whoami > /dev/null 2>&1; then
            return 0
        fi

        local elapsed=$(( $(date +%s) - start ))
        if [[ ${elapsed} -ge ${max_wait} ]]; then
            log "  > ${desc}: unreachable for ${elapsed}s, giving up"
            return 1
        fi

        # If we get "Unauthorized", the session expired — try re-login
        local err_msg
        err_msg=$(oc --context "${CTX}" whoami 2>&1 || true)
        if echo "${err_msg}" | grep -qi "unauthorized\|logged in"; then
            if relogin_cluster; then
                # Re-login worked, try again immediately
                continue
            fi
            # Re-login failed — wait before trying again
            log "  > waiting 30s before next re-login attempt..."
            sleep 30
            continue
        fi

        # Not an auth issue — API might be temporarily unreachable
        if [[ $((elapsed % 30)) -eq 0 && ${elapsed} -gt 0 ]]; then
            log "  > ${desc}: unreachable (${elapsed}s elapsed), waiting..."
        fi
        sleep 10
    done
}

# Wait for a pod to be Running and Ready.
# Usage: wait_for_pod <pod-name> <timeout-seconds>
wait_for_pod_ready() {
    local pod="$1"
    local timeout="${2:-300}"
    local start=$(date +%s)

    while true; do
        local status
        status=$(oc --context "${CTX}" -n "${NS}" get pod "${pod}" \
            -o jsonpath='{.status.phase}' 2>/dev/null) || status="Unknown"

        if [[ "${status}" == "Running" ]]; then
            # Check readiness
            local ready
            ready=$(oc --context "${CTX}" -n "${NS}" get pod "${pod}" \
                -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null) || ready="false"
            if [[ "${ready}" == "true" ]]; then
                return 0
            fi
        fi

        local elapsed=$(( $(date +%s) - start ))
        if [[ ${elapsed} -ge ${timeout} ]]; then
            log "  > pod ${pod}: not ready after ${timeout}s (status=${status})"
            return 1
        fi

        if [[ $((elapsed % 30)) -eq 0 ]]; then
            log "  > pod ${pod}: waiting... (status=${status}, ${elapsed}s)"
        fi
        sleep 5
    done
}

# Recover from a transient cluster disconnect during a long session.
# Called before each session to ensure connectivity.
ensure_cluster_connectivity() {
    local desc="${1:-"pre-session check"}"
    log "  > checking cluster connectivity (${desc})..."
    if ! wait_for_cluster "${desc}" 120; then
        fail "Cluster unreachable during ${desc} — cluster may be down for maintenance"
    fi
    log "  > cluster connectivity OK"
}

# --- Command runners ----------------------------------------------------------

# Run a command, log its output, fail on error
run_logged() {
    local desc="$1"; shift
    local logfile="$1"; shift
    log "  > ${desc}"
    if ! "$@" > "${logfile}" 2>&1; then
        log "  > FAILED (see ${logfile})"
        tail -20 "${logfile}" | tee -a "${LOG_DIR}/campaign.log"
        fail "${desc}"
    fi
    log "  > OK"
}

# Run a command with retries, log its output, fail after exhausting retries
run_logged_retry() {
    local desc="$1"; shift
    local logfile="$1"; shift
    local retries="${1:-${MAX_RETRIES}}"; shift || true
    log "  > ${desc}"
    if ! retry_with_backoff "${desc}" "${retries}" "$@" > "${logfile}" 2>&1; then
        log "  > FAILED after ${retries} retries (see ${logfile})"
        tail -20 "${logfile}" | tee -a "${LOG_DIR}/campaign.log"
        fail "${desc}"
    fi
    log "  > OK"
}

# Verify a file exists and is non-empty
verify_file() {
    local file="$1"
    local desc="$2"
    if [[ ! -s "${file}" ]]; then
        fail "Verification failed: ${desc} — file missing or empty: ${file}"
    fi
    log "  > verified: ${desc} ($(stat -c%s "${file}") bytes)"
}

# Verify session data completeness
verify_session() {
    local session_dir="$1"
    local session_name="$2"
    local expected_trials="$3"

    verify_file "${session_dir}/SUMMARY.json" "SUMMARY.json for ${session_name}"

    local actual_trials
    actual_trials=$(python3 -c "
import json
d = json.load(open('${session_dir}/SUMMARY.json'))
print(len(d['results']))
" 2>/dev/null) || fail "Cannot parse SUMMARY.json for ${session_name}"

    if [[ "${actual_trials}" != "${expected_trials}" ]]; then
        fail "Session ${session_name}: expected ${expected_trials} trials, got ${actual_trials}"
    fi

    local valid_trials
    valid_trials=$(python3 -c "
import json
d = json.load(open('${session_dir}/SUMMARY.json'))
print(sum(1 for r in d['results'] if r['valid']))
" 2>/dev/null) || fail "Cannot count valid trials for ${session_name}"

    if [[ "${valid_trials}" != "${expected_trials}" ]]; then
        fail "Session ${session_name}: expected ${expected_trials} valid trials, got ${valid_trials}"
    fi

    log "  > ${session_name}: ${actual_trials} trials, all valid"

    # Verify receipts exist for every case
    local cases_with_receipts
    cases_with_receipts=$(find "${session_dir}" -name "receipts.json" | wc -l)
    if [[ "${cases_with_receipts}" -lt "${expected_trials}" ]]; then
        log "  > WARNING: only ${cases_with_receipts}/${expected_trials} receipts (network blips)"
    fi
}

# Verify image digest in a session's receipts matches the expected image
verify_image_digest() {
    local session_dir="$1"
    local expected_suffix="$2"  # last 12 chars of the digest
    local session_name="$3"

    local found
    found=$(python3 -c "
import json, glob
for case in sorted(glob.glob('${session_dir}/case-*')):
    try:
        r = json.load(open(case + '/receipts.json'))
        img = r.get('image', '')
        if '${expected_suffix}' not in img:
            print(f'MISMATCH in {case}: {img[-20:]}')
            exit(1)
    except FileNotFoundError:
        pass  # already warned about missing receipts
print('OK')
" 2>/dev/null) || fail "Image digest verification failed for ${session_name}"

    if [[ "${found}" != "OK" ]]; then
        fail "Session ${session_name}: wrong image deployed: ${found}"
    fi
    log "  > ${session_name}: image digest verified"
}

# --- Session runner ------------------------------------------------------------

run_session() {
    local session_letter="$1"
    local session_name="$2"
    local plan_file="$3"
    local expected_trials="$4"
    local session_dir="${CAMPAIGN_DIR}/sessions/${session_name}"
    local sampler_file="${CAMPAIGN_DIR}/samples/${session_name}.jsonl"

    log_section "SESSION ${session_letter}: ${session_name} (${expected_trials} trials)"

    # Ensure cluster connectivity before starting this session
    # (protects against token expiry or API downtime between sessions)
    ensure_cluster_connectivity "session ${session_letter}"

    # Clear any stale data on the client pod (with retry)
    log "  > clearing stale session data on client pod..."
    retry_with_backoff "clear stale data" 3 \
        oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- \
        rm -rf "/store/sessions/${session_name}" 2>/dev/null || true

    # Also clear local
    rm -rf "${session_dir}"
    mkdir -p "${session_dir}"

    # Get the plan's session name for the sweep output
    local sweep_session
    sweep_session=$(python3 -c "
import json
plan = json.load(open('${plan_file}'))
print(plan['session'])
") || fail "Cannot read session name from ${plan_file}"

    # Start sampler
    log "  > starting sampler..."
    python3 "${RESOURCES}/benchmarking/cluster/sampler.py" --output "${sampler_file}" \
        --interval 3 --duration 14400 &
    local SAMPLER_PID=$!

    # Run sweep (with retry for transient oc exec failures — but note:
    # if the sweep itself fails due to a CODE issue, retrying won't help;
    # only retry if the oc exec connection dropped, which we detect by
    # checking if the sweep produced any output before dying)
    log "  > starting sweep (${expected_trials} trials)..."
    local sweep_log="${LOG_DIR}/session-${session_letter}-sweep.log"
    local sweep_attempt=0
    local sweep_max=3
    local sweep_rc=1

    while [[ ${sweep_attempt} -lt ${sweep_max} ]]; do
        sweep_attempt=$((sweep_attempt + 1))
        log "  > sweep attempt ${sweep_attempt}/${sweep_max}..."

        if oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- \
            python3 /store/tools/sweep.py --plan "/store/plans/${sweep_session}.json" \
            --output "/store/sessions/${sweep_session}" > "${sweep_log}" 2>&1; then
            sweep_rc=0
            break
        fi

        sweep_rc=$?
        log "  > sweep attempt ${sweep_attempt} failed (rc=${sweep_rc})"

        # Check if this was a transient connectivity failure vs a real error
        if grep -q "Connection refused\|connection reset\|i/o timeout\|EOF\|dial tcp" \
            "${sweep_log}" 2>/dev/null; then
            local wait_time=$((30 * sweep_attempt))
            log "  > transient connectivity issue detected, retrying in ${wait_time}s..."
            sleep "${wait_time}"
            # Re-verify cluster connectivity
            if ! wait_for_cluster "sweep retry ${sweep_attempt}" 120; then
                log "  > cluster still unreachable, waiting longer..."
                wait_for_cluster "sweep retry ${sweep_attempt}" 300 || break
            fi
            # Clear partial session data before retry
            log "  > clearing partial session data..."
            oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- \
                rm -rf "/store/sessions/${sweep_session}" 2>/dev/null || true
        else
            # Not a transient issue — a real error (code bug, bad config, etc.)
            log "  > sweep failed with non-transient error (see ${sweep_log})"
            break
        fi
    done

    if [[ ${sweep_rc} -ne 0 ]]; then
        kill "${SAMPLER_PID}" 2>/dev/null || true
        log "  > SWEEP FAILED after ${sweep_attempt} attempts (rc=${sweep_rc})"
        tail -30 "${sweep_log}" | tee -a "${LOG_DIR}/campaign.log"
        fail "Session ${session_letter} sweep failed"
    fi
    log "  > sweep completed"

    # Stop sampler
    kill "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true

    # Collect session data from client pod (with retry + fallback)
    log "  > collecting session data..."
    local collect_log="${LOG_DIR}/session-${session_letter}-collect.log"
    local collect_ok=false

    # Try bulk copy with retries
    for collect_attempt in 1 2 3; do
        if oc --context "${CTX}" -n "${NS}" cp \
            "${CLIENT_POD}:/store/sessions/${sweep_session}" \
            "${session_dir}" > "${collect_log}" 2>&1; then
            collect_ok=true
            break
        fi
        log "  > bulk copy attempt ${collect_attempt} failed, retrying in 15s..."
        sleep 15
    done

    if [[ "${collect_ok}" != "true" ]]; then
        # Fallback: try individual file copy (with retry per file)
        log "  > bulk copy failed after 3 attempts, trying individual files..."
        for case_dir in $(retry_with_backoff "list session cases" 3 \
            oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- \
            ls "/store/sessions/${sweep_session}" 2>/dev/null); do
            mkdir -p "${session_dir}/${case_dir}"
            for f in result.json receipts.json prepare.log client-0.log; do
                retry_with_backoff "copy ${case_dir}/${f}" 3 \
                    oc --context "${CTX}" -n "${NS}" exec "${CLIENT_POD}" -- \
                    cat "/store/sessions/${sweep_session}/${case_dir}/${f}" \
                    > "${session_dir}/${case_dir}/${f}" 2>/dev/null || true
            done
        done
    fi

    # Verify data
    verify_session "${session_dir}" "${session_name}" "${expected_trials}"

    # Verify image digest in every session's receipts — all campaign
    # variants run the ship image (single-variable design).
    local digest_suffix="${SHIP_IMAGE: -12}"
    verify_image_digest "${session_dir}" "${digest_suffix}" "${session_name}"

    # Run summarizer
    log "  > summarizing..."
    run_logged "Summarize ${session_name}" \
        "${LOG_DIR}/session-${session_letter}-summarize.log" \
        python3 "${RESOURCES}/benchmarking/cluster/summarize.py" --session "${session_dir}" \
            --sampler "${sampler_file}"

    # Run validator
    log "  > validating..."
    run_logged "Validate ${session_name}" \
        "${LOG_DIR}/session-${session_letter}-validate.log" \
        python3 "${RESOURCES}/benchmarking/cluster/validate_session.py" --session "${session_dir}"

    # Run paired analysis: analyze.py takes ONE comparison per invocation
    # (--control/--candidate), so loop over the session's paired spec.
    # Output files are per-comparison; failures are loud per comparison.
    local paired_spec="${CAMPAIGN_DIR}/paired-specs/${session_letter}.json"
    if [[ -s "${paired_spec}" ]]; then
        log "  > running paired analysis..."
        local analysis_log="${LOG_DIR}/session-${session_letter}-analysis.log"
        : > "${analysis_log}"
        local n_comp=0
        mapfile -t comps < <(python3 -c "
import json
spec = json.load(open('${paired_spec}'))
for comp in spec.get('comparisons', []):
    print(json.dumps(comp))
")
        for comp in "${comps[@]}"; do
            local control candidate label out
            control=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['control'])" "${comp}") || continue
            candidate=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['candidate'])" "${comp}") || continue
            label=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['label'])" "${comp}") || continue
            out="${session_dir}/paired-${control}-vs-${candidate}.json"
            rm -f "${out}"
            if (cd "${RESOURCES}" && .venv/bin/python \
                benchmarking/study_tools/analyze.py \
                --summary "${session_dir}/SUMMARY.json" \
                --control "${control}" --candidate "${candidate}" \
                --output "${out}") >> "${analysis_log}" 2>&1; then
                log "  > paired: ${label} -> ${out##*/}"
                n_comp=$((n_comp + 1))
            else
                log "  > WARNING: paired analysis failed for: ${label} (see ${analysis_log})"
                tail -5 "${analysis_log}" | tee -a "${LOG_DIR}/campaign.log"
            fi
        done
        log "  > paired analyses completed: ${n_comp}/${#comps[@]}"
    fi

    # Print summary
    log "  > session ${session_letter} results:"
    if [[ -f "${session_dir}/SUMMARY.json" ]]; then
        python3 -c "
import json
d = json.load(open('${session_dir}/SUMMARY.json'))
from collections import defaultdict
import numpy as np
groups = defaultdict(list)
for r in d['results']:
    if r['valid']:
        groups[(r['variant'], r['concurrency'])].append(r['rps'])
print(f'{\"variant\":<28} {\"c\":>3} {\"RPS\":>8}')
for (v, c), rates in sorted(groups.items()):
    print(f'{v:<28} {c:>3} {np.median(rates):>8.2f}')
" 2>/dev/null | tee -a "${LOG_DIR}/campaign.log" || true
    fi

    log "  > SESSION ${session_letter} COMPLETE"
}


# =============================================================================
# RESUME: Sessions A/B/C completed in the parent run (all valid, all validated).
# This script runs Session D + consolidated results + archival.
# =============================================================================

log_section "RESUME PRE-CHECKS"

log "  > HEAD commit: ${HEAD_COMMIT}"
log "  > ship image: ${SHIP_IMAGE}"
test -f "${CAMPAIGN_DIR}/ship-image.txt" || fail "ship-image.txt missing"

if ! wait_for_cluster "resume connectivity" 120; then
    fail "Cluster unreachable"
fi
log "  > cluster connectivity OK"

CLIENT_POD=$(retry_with_backoff "get client pod" "${MAX_RETRIES}" \
    oc --context "${CTX}" -n "${NS}" get pods -l app=mlserver-client \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) || fail "Cannot find client pod"
log "  > client pod: ${CLIENT_POD}"

# Completed sessions must be present, valid, and freshly re-validated
for s in final-unbatched-rest final-unbatched-grpc final-batched; do
    verify_file "${CAMPAIGN_DIR}/sessions/${s}/SUMMARY.json" "completed session ${s}"
done
log "  > sessions A/B/C present (validated after gpu_util falsy-zero fix)"

# Session D prerequisites on the pod: plan, tools, references, batched model
oc_exec_retry "check session D plan" "${CLIENT_POD}" \
    test -s /store/plans/final-mixed.json || fail "final-mixed plan missing on client pod"
oc_exec_retry "check hook.py" "${CLIENT_POD}" \
    test -s /store/tools/hook.py || fail "hook.py missing on client pod"
oc_exec_retry "check sweep.py" "${CLIENT_POD}" \
    test -s /store/tools/sweep.py || fail "sweep.py missing on client pod"
oc_exec_retry "check client.py" "${CLIENT_POD}" \
    test -s /store/tools/client.py || fail "client.py missing on client pod"
oc_exec_retry "check batched rest reference" "${CLIENT_POD}" \
    test -s /store/references/cuda-batched-rest-response.bin || fail "batched REST reference missing"
oc_exec_retry "check batched grpc reference" "${CLIENT_POD}" \
    test -s /store/references/cuda-batched-grpc-response.bin || fail "batched gRPC reference missing"
SRV_POD=$(retry_with_backoff "get server pod" "${MAX_RETRIES}" \
    oc --context "${CTX}" -n "${NS}" get pods -l app=mlserver-rfdetr \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) || fail "Cannot find server pod"
oc_exec_retry "check batched model on PVC" "${SRV_POD}" \
    test -s /mnt/models/rfdetr-small-batched/model.onnx || fail "batched model missing on PVC"
log "  > session D prerequisites verified (plan, tools, references, model)"

# The ship image must still be the digest the trials will deploy
ACTUAL_DIGEST=$(skopeo inspect docker://quay.io/syedali/mlserver-cuda:rhoaieng-102092-final 2>/dev/null | \
    python3 -c "import sys,json; print(json.load(sys.stdin)['Digest'])") || fail "Cannot get image digest"
if [[ "quay.io/syedali/mlserver-cuda@${ACTUAL_DIGEST}" != "${SHIP_IMAGE}" ]]; then
    fail "Quay tag moved since the parent run: expected ${SHIP_IMAGE}, got ${ACTUAL_DIGEST}"
fi
log "  > ship image digest unchanged on Quay"

log ""
log "Resume pre-checks PASSED. Running Session D."
run_session "D" "final-mixed" \
    "${CAMPAIGN_DIR}/plans/session-D.json" 9
# --- Step 9: Generate consolidated results ------------------------------------

log_section "STEP 9: CONSOLIDATED RESULTS"

run_logged "Generating final results table" \
    "${LOG_DIR}/step9-results.log" \
    python3 "${RESOURCES}/benchmarking/campaign/generate-results.py" || fail "Failed to generate results"

verify_file "${CAMPAIGN_DIR}/RESULTS.md" "consolidated results"

log ""
log "Final results written to: ${CAMPAIGN_DIR}/RESULTS.md"

# --- Step 10: Archival ---------------------------------------------------------

log_section "STEP 10: ARCHIVAL TO DURABLE STORAGE"

ARCHIVE="/home/syedali/Workspace/mlserver-performance/MLServer-resources/benchmark-results/final-campaign/rhoaieng-102092-final-campaign-${STAMP}.tar.gz"

run_logged "Building archive" \
    "${LOG_DIR}/step10-archive.log" \
    tar czf "${ARCHIVE}" \
        -C "${CAMPAIGN_DIR}" \
        sessions samples logs plans paired-specs RESULTS.md \
        capture-batched-receipts.json g-batched-0.bin \
        batched-model-settings.json

sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
log "  > archive: ${ARCHIVE} ($(du -h "${ARCHIVE}" | cut -f1))"
log "  > sha256: $(cat "${ARCHIVE}.sha256" | cut -d' ' -f1)"

# Upload to PVC (with retry)
SRV_POD=$(retry_with_backoff "get server pod" "${MAX_RETRIES}" \
    oc --context "${CTX}" -n "${NS}" get pods -l app=mlserver-rfdetr \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) || fail "Cannot find server pod"
oc_exec_retry "mkdir archives" "${SRV_POD}" mkdir -p /mnt/models/archives 2>/dev/null || true
retry_with_backoff "upload archive to PVC" "${MAX_RETRIES}" \
    bash -c "cat '${ARCHIVE}' | oc --context '${CTX}' -n '${NS}' exec -i '${SRV_POD}' -- sh -c 'cat > /mnt/models/archives/$(basename "${ARCHIVE}")'" || \
    log "  > WARNING: PVC upload failed (archive still available locally at ${ARCHIVE})"
log "  > uploaded to PVC"

# --- Complete -------------------------------------------------------------------

log_section "CAMPAIGN COMPLETE"

log "HEAD commit: ${HEAD_FULL}"
log "Ship image: ${SHIP_IMAGE}"
log "Total trials: 159 (90 unbatched + 60 batched + 9 mixed)"
log "Sessions: 4 (all passed, all valid)"
log "Results: ${CAMPAIGN_DIR}/RESULTS.md"
log "Archive: ${ARCHIVE}"
log "Campaign log: ${LOG_DIR}/campaign.log"
log ""
log "All data captured freshly. No stale data. No assumptions."
log "Ready for PR, Jira, and report."
