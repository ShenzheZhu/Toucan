#!/usr/bin/env bash
# Automated end-to-end Toucan data generation pipeline using OpenAI engines.
# Each step pauses for confirmation and records a checkpoint so the pipeline
# can resume after failures. To resume, re-run this script with the same JOB_ID.

set -euo pipefail

confirm_or_exit() {
  local prompt="$1"
  echo "$prompt"
}

find_latest_file() {
  local pattern=$1
  ls -1t ${pattern} 2>/dev/null | head -n1 || true
}

run_step3_agent_with_monitor() {
  local results_pattern=$1
  shift
  local autokill="${STEP3_AUTOKILL_ON_RESULTS:-true}"
  local interval="${STEP3_MONITOR_INTERVAL:-15}"

  (
    cd datagen && \
    "$@"
  ) &
  local agent_pid=$!
  local detected_file=""

  echo "Step 3.1 agent started with PID ${agent_pid} (autokill=${autokill}, interval=${interval}s)"

  while kill -0 "${agent_pid}" 2>/dev/null; do
    detected_file=$(find_latest_file "${results_pattern}")
    if [[ -n "${detected_file}" && -s "${detected_file}" ]]; then
      echo "Detected Step 3.1 results file: ${detected_file}"
      if [[ "${autokill}" == "true" ]]; then
        echo "Auto-stopping agent PID ${agent_pid} now that results exist..."
        kill "${agent_pid}" 2>/dev/null || true
      fi
      wait "${agent_pid}" 2>/dev/null || true
      AGENT_RESULTS="${detected_file}"
      return 0
    fi
    sleep "${interval}"
  done

  wait "${agent_pid}" 2>/dev/null || true
  detected_file=$(find_latest_file "${results_pattern}")
  if [[ -n "${detected_file}" ]]; then
    AGENT_RESULTS="${detected_file}"
    return 0
  fi
  return 1
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

TOTAL_PROMPTS=${TOTAL_PROMPTS:-10}
SAMPLING_STRATEGY=${SAMPLING_STRATEGY:-random}
NUM_TOOLS=${NUM_TOOLS:-}
MODE=${MODE:-single_server}
POWER_LAW_ALPHA=${POWER_LAW_ALPHA:-}
MCP_SOURCE=${MCP_SOURCE:-mcp_new}
SAMPLES_PER_SERVER=${SAMPLES_PER_SERVER:-}

QUESTION_MODEL=${QUESTION_MODEL:-gpt-5-mini}
QUESTION_ENGINE=${QUESTION_ENGINE:-openai}
QUESTION_START_VLLM=${QUESTION_START_VLLM:-false}
QUESTION_BATCH_SIZE=${QUESTION_BATCH_SIZE:-5}
QUESTION_MAX_TOKENS=${QUESTION_MAX_TOKENS:-4096}

QC_MODEL=${QC_MODEL:-gpt-5-mini}
QC_ENGINE=${QC_ENGINE:-openai}
QC_START_VLLM=${QC_START_VLLM:-false}
QC_BATCH_SIZE=${QC_BATCH_SIZE:-5}
QC_MAX_TOKENS=${QC_MAX_TOKENS:-2048}

AGENT_MODEL=${AGENT_MODEL:-gpt-5-mini}
AGENT_ENGINE=${AGENT_ENGINE:-openai}
AGENT_AGENT=${AGENT_AGENT:-openai_agent}
AGENT_START_VLLM=${AGENT_START_VLLM:-false}
AGENT_BATCH_SIZE=${AGENT_BATCH_SIZE:-2}
AGENT_MAX_TOKENS=${AGENT_MAX_TOKENS:-4096}
AGENT_LIMIT_TO_TARGET_TOOLS=${AGENT_LIMIT_TO_TARGET_TOOLS:-false}
SMITHERY_API_POOL_FILE=${SMITHERY_API_POOL_FILE:-${PROJECT_ROOT}/smithery_api_pool.json}

RESPONSE_QC_MODEL=${RESPONSE_QC_MODEL:-gpt-5-mini}
RESPONSE_QC_ENGINE=${RESPONSE_QC_ENGINE:-openai}
RESPONSE_QC_START_VLLM=${RESPONSE_QC_START_VLLM:-false}
RESPONSE_QC_BATCH_SIZE=${RESPONSE_QC_BATCH_SIZE:-5}
RESPONSE_QC_MAX_TOKENS=${RESPONSE_QC_MAX_TOKENS:-2048}

JOB_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_SOURCE_TAG=${MCP_SOURCE:-mcp_new}
JOB_MODE_TAG=${INTERACTION_MODE:-single_turn}
JOB_ID=${JOB_ID:-ToolUse_job_${JOB_TIMESTAMP}-${JOB_SOURCE_TAG}-${JOB_MODE_TAG}}
JOB_DIR=${JOB_DIR:-data/${JOB_ID}}
mkdir -p "${JOB_DIR}"

CHECKPOINT_DIR="${JOB_DIR}/.checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

step_file() { echo "${CHECKPOINT_DIR}/${1}.txt"; }
step_completed() { [[ -f "$(step_file "$1")" ]]; }
set_checkpoint() { echo "$2" > "$(step_file "$1")"; }
get_checkpoint() { local f; f=$(step_file "$1"); [[ -f "${f}" ]] && cat "${f}"; }

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set."
  exit 1
fi

echo "================ Toucan OpenAI Pipeline ================"
echo " Job ID           : ${JOB_ID}"
echo " Output directory : ${JOB_DIR}"
echo " Sampling         : ${SAMPLING_STRATEGY} (total_prompts=${TOTAL_PROMPTS})"
echo " MCP source       : ${MCP_SOURCE}"
echo " Question model   : ${QUESTION_MODEL}"
echo " QC model         : ${QC_MODEL}"
echo " Agent model      : ${AGENT_MODEL}"
echo " Agent limit to targets : ${AGENT_LIMIT_TO_TARGET_TOOLS}"
echo " Response QC model: ${RESPONSE_QC_MODEL}"
echo "========================================================"

# ---------------- Step 1.1 ----------------
if step_completed "step1_prompts"; then
  STEP1_OUTPUT=$(get_checkpoint "step1_prompts")
  echo "Step 1.1 already completed. Using cached file: ${STEP1_OUTPUT}"
else
  confirm_or_exit "Step 1.1 will generate prompts. Continue?"
  cmd=(python datagen/step1.1_gen_questions.py \
    --job_name "${JOB_ID}" \
    --mcp_source "${MCP_SOURCE}" \
    --output_folder "${JOB_DIR}")
  if [[ -n "${TOTAL_PROMPTS}" ]]; then
    cmd+=(--total_prompts "${TOTAL_PROMPTS}")
  fi
  if [[ -n "${SAMPLING_STRATEGY}" ]]; then
    cmd+=(--sampling_strategy "${SAMPLING_STRATEGY}")
  fi
  if [[ -n "${NUM_TOOLS}" ]]; then
    cmd+=(--num_tools "${NUM_TOOLS}")
  fi
  if [[ -n "${MODE}" ]]; then
    cmd+=(--mode "${MODE}")
  fi
  if [[ -n "${POWER_LAW_ALPHA}" ]]; then
    cmd+=(--power_law_alpha "${POWER_LAW_ALPHA}")
  fi
  if [[ -n "${SAMPLES_PER_SERVER}" ]]; then
    cmd+=(--samples_per_server "${SAMPLES_PER_SERVER}")
  fi
  "${cmd[@]}"
  STEP1_OUTPUT=$(find_latest_file "${JOB_DIR}/ToolUse_s2q_*_prepared.jsonl")
  if [[ -z "${STEP1_OUTPUT}" || ! -f "${STEP1_OUTPUT}" ]]; then
    echo "ERROR: Step 1.1 output not found."
    exit 1
  fi
  set_checkpoint "step1_prompts" "${STEP1_OUTPUT}"
fi
echo "Step 1.1 output: ${STEP1_OUTPUT}"

# ---------------- Step 1.2 ----------------
if step_completed "step1_completions"; then
  STEP1_RESULTS=$(get_checkpoint "step1_completions")
  echo "Step 1.2 already completed. Using cached file: ${STEP1_RESULTS}"
else
  confirm_or_exit "Step 1.2 will generate question completions. Continue?"
  QUESTION_API_KEY="${OPENAI_API_KEY:-}"
  if [[ "${QUESTION_ENGINE}" != "openai" ]]; then
    QUESTION_API_KEY=""
  fi
  (
    cd datagen && \
    bash step1.2_completion.sh \
      "${PROJECT_ROOT}/${STEP1_OUTPUT}" \
      "${QUESTION_MODEL}" \
      "${QUESTION_ENGINE}" \
      1.2 \
      "${QUESTION_START_VLLM}" \
      "${QUESTION_API_KEY}" \
      --batch_size "${QUESTION_BATCH_SIZE}" \
      --max_tokens "${QUESTION_MAX_TOKENS}"
  )
  STEP1_BASE_NO_EXT="${STEP1_OUTPUT%.jsonl}"
  STEP1_SEARCH_BASE="${STEP1_BASE_NO_EXT}"
  if [[ "${STEP1_SEARCH_BASE}" == *_4prepared ]]; then
    STEP1_SEARCH_BASE="${STEP1_SEARCH_BASE%_4prepared}"
  elif [[ "${STEP1_SEARCH_BASE}" == *_prepared ]]; then
    STEP1_SEARCH_BASE="${STEP1_SEARCH_BASE%_prepared}"
  fi
  STEP1_RESULTS=$(find_latest_file "$(dirname "${STEP1_OUTPUT}")/$(basename "${STEP1_SEARCH_BASE}")_*_results.jsonl")
  if [[ -z "${STEP1_RESULTS}" || ! -f "${STEP1_RESULTS}" ]]; then
    echo "ERROR: Step 1.2 results not found."
    exit 1
  fi
  set_checkpoint "step1_completions" "${STEP1_RESULTS}"
fi
echo "Step 1.2 results: ${STEP1_RESULTS}"

# ---------------- Step 1.3 ----------------
if step_completed "step1_sanitized"; then
  SANITIZED=$(get_checkpoint "step1_sanitized")
  echo "Step 1.3 already completed. Using cached file: ${SANITIZED}"
else
  confirm_or_exit "Step 1.3 will sanitize questions. Continue?"
  python datagen/step1.3_process_completion.py --input_file "${STEP1_RESULTS}"
  STEP1_BASE_NO_EXT="${STEP1_OUTPUT%.jsonl}"
  STEP1_SEARCH_BASE="${STEP1_BASE_NO_EXT}"
  if [[ "${STEP1_SEARCH_BASE}" == *_4prepared ]]; then
    STEP1_SEARCH_BASE="${STEP1_SEARCH_BASE%_4prepared}"
  elif [[ "${STEP1_SEARCH_BASE}" == *_prepared ]]; then
    STEP1_SEARCH_BASE="${STEP1_SEARCH_BASE%_prepared}"
  fi
  SANITIZED=$(find_latest_file "$(dirname "${STEP1_OUTPUT}")/processed/$(basename "${STEP1_SEARCH_BASE}")_*_3sanitized.jsonl")
  if [[ -z "${SANITIZED}" || ! -f "${SANITIZED}" ]]; then
    echo "ERROR: Step 1.3 sanitized output not found."
    exit 1
  fi
  set_checkpoint "step1_sanitized" "${SANITIZED}"
fi
echo "Sanitized questions: ${SANITIZED}"

# ---------------- Step 2.1 ----------------
if step_completed "step2_prompts"; then
  STEP2_PROMPTS=$(get_checkpoint "step2_prompts")
  echo "Step 2.1 already completed. Using cached file: ${STEP2_PROMPTS}"
else
  confirm_or_exit "Step 2.1 will create QC prompts. Continue?"
  (
    cd datagen && \
    python step2.1_question_quality_check.py --input_file "${PROJECT_ROOT}/${SANITIZED}"
  )
  STEP2_BASE=${SANITIZED%.jsonl}
  STEP2_PROMPTS="${STEP2_BASE}_qced_prepared.jsonl"
  if [[ ! -f "${STEP2_PROMPTS}" ]]; then
    STEP2_PROMPTS=$(find_latest_file "$(dirname "${SANITIZED}")/$(basename "${STEP2_BASE}")*_qced_prepared.jsonl")
  fi
  if [[ -z "${STEP2_PROMPTS}" || ! -f "${STEP2_PROMPTS}" ]]; then
    echo "ERROR: Step 2.1 output not found."
    exit 1
  fi
  set_checkpoint "step2_prompts" "${STEP2_PROMPTS}"
fi
echo "Step 2.1 output: ${STEP2_PROMPTS}"

# ---------------- Step 2.2 ----------------
if step_completed "step2_completions"; then
  STEP2_RESULTS=$(get_checkpoint "step2_completions")
  echo "Step 2.2 already completed. Using cached file: ${STEP2_RESULTS}"
else
  confirm_or_exit "Step 2.2 will run question QC completions. Continue?"
  QC_API_KEY="${OPENAI_API_KEY:-}"
  if [[ "${QC_ENGINE}" != "openai" ]]; then
    QC_API_KEY=""
  fi
  (
    cd datagen && \
    bash step2.2_completion_quality_check.sh \
    "${PROJECT_ROOT}/${STEP2_PROMPTS}" \
    "${QC_MODEL}" \
    "${QC_ENGINE}" \
    2.2 \
    "${QC_START_VLLM}" \
    "${QC_API_KEY}" \
    --batch_size "${QC_BATCH_SIZE}" \
    --max_tokens "${QC_MAX_TOKENS}"
  )
  STEP2_BASE_NO_EXT="${STEP2_PROMPTS%.jsonl}"
  STEP2_SEARCH_BASE="${STEP2_BASE_NO_EXT}"
  if [[ "${STEP2_SEARCH_BASE}" == *_qced_prepared ]]; then
    STEP2_SEARCH_BASE="${STEP2_SEARCH_BASE%_qced_prepared}"
  elif [[ "${STEP2_SEARCH_BASE}" == *_prepared ]]; then
    STEP2_SEARCH_BASE="${STEP2_SEARCH_BASE%_prepared}"
  fi
  STEP2_RESULTS=$(find_latest_file "$(dirname "${STEP2_PROMPTS}")/$(basename "${STEP2_SEARCH_BASE}")_*_results.jsonl")
  if [[ -z "${STEP2_RESULTS}" || ! -f "${STEP2_RESULTS}" ]]; then
    echo "ERROR: Step 2.2 results not found."
    exit 1
  fi
  set_checkpoint "step2_completions" "${STEP2_RESULTS}"
fi
echo "Step 2.2 results: ${STEP2_RESULTS}"

# ---------------- Step 2.3 ----------------
if step_completed "step2_aggregation"; then
  QUALITY_PREPARED=$(get_checkpoint "step2_aggregation")
  echo "Step 2.3 already completed. Using cached file: ${QUALITY_PREPARED}"
else
  confirm_or_exit "Step 2.3 will aggregate QC outputs. Continue?"
  python datagen/step2.3_process_completion.py --input_file "${STEP2_RESULTS}"
  STEP2_RESULTS_BASE="${STEP2_RESULTS%.jsonl}"
  STEP2_RESULTS_TRIM="${STEP2_RESULTS_BASE}"
  if [[ "${STEP2_RESULTS_TRIM}" == *_results ]]; then
    STEP2_RESULTS_TRIM="${STEP2_RESULTS_TRIM%_results}"
  fi
  QUALITY_PREPARED=$(find_latest_file "$(dirname "${SANITIZED}")/quality_checked/$(basename "${STEP2_RESULTS_TRIM}")_2prepared.jsonl")
  if [[ -z "${QUALITY_PREPARED}" || ! -f "${QUALITY_PREPARED}" ]]; then
    echo "ERROR: Step 2.3 output not found."
    exit 1
  fi
  set_checkpoint "step2_aggregation" "${QUALITY_PREPARED}"
fi
echo "Quality-checked prompts: ${QUALITY_PREPARED}"

# ---------------- Step 3.1 ----------------
if step_completed "step3_agent"; then
  AGENT_RESULTS=$(get_checkpoint "step3_agent")
  echo "Step 3.1 already completed. Using cached file: ${AGENT_RESULTS}"
else
  confirm_or_exit "Step 3.1 will launch the agent. Continue?"
  STEP3_EXTRA=()
  [[ -n "${SMITHERY_API_KEY:-}" ]] && STEP3_EXTRA+=(--smithery_api_key "${SMITHERY_API_KEY}")
  [[ -n "${SMITHERY_PROFILE:-}" ]] && STEP3_EXTRA+=(--smithery_profile "${SMITHERY_PROFILE}")
  [[ -n "${SMITHERY_API_POOL_FILE:-}" && -f "${SMITHERY_API_POOL_FILE}" ]] && STEP3_EXTRA+=(--smithery_api_pool "${SMITHERY_API_POOL_FILE}")
  AGENT_MAX_TURNS=${AGENT_MAX_TURNS:-}
  if [[ -n "${AGENT_MAX_TURNS}" ]]; then
    STEP3_EXTRA+=(--max_turns "${AGENT_MAX_TURNS}")
  fi
  if [[ -n "${AGENT_REASONING_EFFORT:-}" ]]; then
    STEP3_EXTRA+=(--reasoning_effort "${AGENT_REASONING_EFFORT}")
  fi
  if [[ -n "${AGENT_TIMEOUT:-}" ]]; then
    STEP3_EXTRA+=(--timeout "${AGENT_TIMEOUT}")
  fi
  if [[ "${AGENT_LIMIT_TO_TARGET_TOOLS}" == "true" ]]; then
    STEP3_EXTRA+=(--limit_tools_to_targets)
  fi
  AGENT_API_KEY="${OPENAI_API_KEY:-}"
  if [[ "${AGENT_ENGINE}" != "openai" ]]; then
    AGENT_API_KEY=""
  fi
  STEP3_RESULTS_PATTERN="$(dirname "${QUALITY_PREPARED}")/$(basename "${QUALITY_PREPARED%.jsonl}")_*_results.jsonl"
  if ! run_step3_agent_with_monitor "${STEP3_RESULTS_PATTERN}" \
    bash step3.1_completion_agent.sh \
    "${PROJECT_ROOT}/${QUALITY_PREPARED}" \
    "${AGENT_MODEL}" \
    "${AGENT_ENGINE}" \
    3.1 \
    "${AGENT_AGENT}" \
    "${AGENT_START_VLLM}" \
    "${AGENT_API_KEY}" \
    --batch_size "${AGENT_BATCH_SIZE}" \
    --max_tokens "${AGENT_MAX_TOKENS}" \
    "${STEP3_EXTRA[@]}"; then
    AGENT_RESULTS=""
  fi
  if [[ -z "${AGENT_RESULTS:-}" ]]; then
    AGENT_RESULTS=$(find_latest_file "${STEP3_RESULTS_PATTERN}")
  fi
  if [[ -z "${AGENT_RESULTS}" || ! -f "${AGENT_RESULTS}" ]]; then
    echo "ERROR: Step 3.1 results not found."
    exit 1
  fi
  set_checkpoint "step3_agent" "${AGENT_RESULTS}"
fi
echo "Agent trajectories: ${AGENT_RESULTS}"

# ---------------- Step 3.2 ----------------
if step_completed "step3_filter"; then
  RULE_FILTERED=$(get_checkpoint "step3_filter")
  echo "Step 3.2 already completed. Using cached file: ${RULE_FILTERED}"
else
  confirm_or_exit "Step 3.2 will filter trajectories. Continue?"
  python datagen/step3.2_process_completion.py --input_file "${AGENT_RESULTS}"
  RULE_FILTERED=$(find_latest_file "$(dirname "${AGENT_RESULTS}")/processed/$(basename "${AGENT_RESULTS%.jsonl}")_rule_filtered.jsonl")
  if [[ -z "${RULE_FILTERED}" || ! -f "${RULE_FILTERED}" ]]; then
    echo "ERROR: Step 3.2 output not found."
    exit 1
  fi
  set_checkpoint "step3_filter" "${RULE_FILTERED}"
fi
echo "Filtered trajectories: ${RULE_FILTERED}"

# ---------------- Step 4.1 ----------------
if step_completed "step4_prompts"; then
  RESP_QC_PROMPTS=$(get_checkpoint "step4_prompts")
  echo "Step 4.1 already completed. Using cached file: ${RESP_QC_PROMPTS}"
else
  confirm_or_exit "Step 4.1 will generate response-QC prompts. Continue?"
  (
    cd datagen && \
    python step4.1_response_quality_check.py --input_file "${PROJECT_ROOT}/${RULE_FILTERED}"
  )
  RESP_QC_PROMPTS="${RULE_FILTERED%.jsonl}_response_qced_prepared.jsonl"
  if [[ ! -f "${RESP_QC_PROMPTS}" ]]; then
    RESP_QC_PROMPTS=$(find_latest_file "$(dirname "${RULE_FILTERED}")/$(basename "${RULE_FILTERED%.jsonl}")*_response_qced_prepared.jsonl")
  fi
  if [[ -z "${RESP_QC_PROMPTS}" || ! -f "${RESP_QC_PROMPTS}" ]]; then
    echo "ERROR: Step 4.1 output not found."
    exit 1
  fi
  set_checkpoint "step4_prompts" "${RESP_QC_PROMPTS}"
fi
echo "Response QC prompts: ${RESP_QC_PROMPTS}"

# ---------------- Step 4.2 ----------------
if step_completed "step4_completions"; then
  RESP_QC_RESULTS=$(get_checkpoint "step4_completions")
  echo "Step 4.2 already completed. Using cached file: ${RESP_QC_RESULTS}"
else
  confirm_or_exit "Step 4.2 will run response QC completions. Continue?"
  RESP_QC_API_KEY="${OPENAI_API_KEY:-}"
  if [[ "${RESPONSE_QC_ENGINE}" != "openai" ]]; then
    RESP_QC_API_KEY=""
  fi
  (
    cd datagen && \
    bash step4.2_completion_response_check.sh \
    "${PROJECT_ROOT}/${RESP_QC_PROMPTS}" \
    "${RESPONSE_QC_MODEL}" \
    "${RESPONSE_QC_ENGINE}" \
    4.2 \
    "${RESPONSE_QC_START_VLLM}" \
    "${RESP_QC_API_KEY}" \
    --batch_size "${RESPONSE_QC_BATCH_SIZE}" \
    --max_tokens "${RESPONSE_QC_MAX_TOKENS}"
  )
  RESP_QC_RESULTS=$(find_latest_file "$(dirname "${RESP_QC_PROMPTS}")/$(basename "${RESP_QC_PROMPTS%.jsonl}")_*_results.jsonl")
  if [[ -z "${RESP_QC_RESULTS}" || ! -f "${RESP_QC_RESULTS}" ]]; then
    echo "ERROR: Step 4.2 results not found."
    exit 1
  fi
  set_checkpoint "step4_completions" "${RESP_QC_RESULTS}"
fi
echo "Response QC results: ${RESP_QC_RESULTS}"

# ---------------- Step 4.3 ----------------
FINAL_OUTPUT_DIR="${JOB_DIR}/completed"
mkdir -p "${FINAL_OUTPUT_DIR}"
if step_completed "step4_aggregation"; then
  FINAL_JSONL=$(get_checkpoint "step4_aggregation")
  echo "Step 4.3 already completed. Using cached file: ${FINAL_JSONL}"
else
  confirm_or_exit "Step 4.3 will aggregate response QC outputs. Continue?"
  python datagen/step4.3_process_completion.py \
    --input_file "${RESP_QC_RESULTS}" \
    --output_folder "${FINAL_OUTPUT_DIR}"
  FINAL_JSONL="${FINAL_OUTPUT_DIR}/$(basename "${RESP_QC_RESULTS%_results.jsonl}")_processed.jsonl"
  set_checkpoint "step4_aggregation" "${FINAL_JSONL}"
fi

# ---------------- Step 5.1 ----------------
NORMALIZED_V3="${JOB_DIR}/all_processed_v3.jsonl"
if step_completed "step5_convert_v3"; then
  NORMALIZED_V3=$(get_checkpoint "step5_convert_v3")
  echo "Step 5.1 already completed. Using cached file: ${NORMALIZED_V3}"
else
  confirm_or_exit "Step 5.1 will normalize final outputs (convert_all_processed). Continue?"
  python data/processed/scripts/convert_all_processed.py \
    --inputs "${FINAL_JSONL}" \
    --output "${NORMALIZED_V3}"
  set_checkpoint "step5_convert_v3" "${NORMALIZED_V3}"
fi
echo "Normalized dataset (v3): ${NORMALIZED_V3}"

# ---------------- Step 5.2 ----------------
MS_SWIFT_OUTPUT="${JOB_DIR}/all_processed_ms_swift.jsonl"
if step_completed "step5_ms_swift"; then
  MS_SWIFT_OUTPUT=$(get_checkpoint "step5_ms_swift")
  echo "Step 5.2 already completed. Using cached file: ${MS_SWIFT_OUTPUT}"
else
  confirm_or_exit "Step 5.2 will convert normalized data to MS-Swift format. Continue?"
  python data/processed/scripts/convert_all_processed_to_ms_swift.py \
    --input "${NORMALIZED_V3}" \
    --output "${MS_SWIFT_OUTPUT}"
  set_checkpoint "step5_ms_swift" "${MS_SWIFT_OUTPUT}"
fi
echo "MS-Swift formatted dataset: ${MS_SWIFT_OUTPUT}"

# ---------------- Step 5.3 ----------------
FINAL_DATASET_PATH="${JOB_DIR}/final_dataset.jsonl"
if [[ -n "${RULE_FILTERED:-}" && -f "${RULE_FILTERED}" ]]; then
  TOOL_POOL_JSON="data/mcp_tool_pool.json"
  if [[ ! -f "${TOOL_POOL_JSON}" ]]; then
    if [[ -x "scripts/cache_mcp_tool_pool.py" ]]; then
      echo "Tool pool cache missing - generating ${TOOL_POOL_JSON} ..."
      if ! ./scripts/cache_mcp_tool_pool.py; then
        echo "⚠️  Failed to generate tool pool cache. Skipping final dataset export."
        TOOL_POOL_JSON=""
      fi
    else
      echo "⚠️  scripts/cache_mcp_tool_pool.py not found. Skipping final dataset export."
      TOOL_POOL_JSON=""
    fi
  fi
  if [[ -n "${TOOL_POOL_JSON}" && -x "scripts/build_final_dataset.py" ]]; then
    echo "Building curated dataset at ${FINAL_DATASET_PATH} ..."
    if ./scripts/build_final_dataset.py \
      --job_id "${JOB_ID}" \
      --input "${RULE_FILTERED}" \
      --output "${FINAL_DATASET_PATH}" \
      --tool_pool "${TOOL_POOL_JSON}"; then
      echo "Final dataset ready: ${FINAL_DATASET_PATH}"
    else
      echo "⚠️  Failed to build final dataset."
    fi
  else
    echo "⚠️  scripts/build_final_dataset.py not executable or tool pool unavailable. Skipping Step 5.3."
  fi
else
  echo "Step 5.3 skipped: RULE_FILTERED file not available."
fi

echo "========================================================"
echo "Pipeline complete!"
echo "  Step1 prompts          : ${STEP1_OUTPUT}"
echo "  Sanitized questions    : ${SANITIZED}"
echo "  QC prepared prompts    : ${STEP2_PROMPTS}"
echo "  Quality-checked input  : ${QUALITY_PREPARED}"
echo "  Filtered trajectories  : ${RULE_FILTERED}"
echo "  Response QC prompts    : ${RESP_QC_PROMPTS}"
echo "  Final processed output : ${FINAL_JSONL}"
echo "  Normalized dataset     : ${NORMALIZED_V3}"
echo "  MS-Swift output        : ${MS_SWIFT_OUTPUT}"
echo "  Final dataset          : ${FINAL_DATASET_PATH}"
echo "Checkpoint files stored in: ${CHECKPOINT_DIR}"
echo "========================================================"
