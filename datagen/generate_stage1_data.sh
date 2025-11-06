#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_SINGLE=${RUN_SINGLE:-1}
RUN_MULTI=${RUN_MULTI:-1}
MCP_SOURCE=${MCP_SOURCE:-mcp_10_minus}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SINGLE_INTERACTION_MODE=${SINGLE_INTERACTION_MODE:-single_turn}
MULTI_INTERACTION_MODE=${MULTI_INTERACTION_MODE:-single_turn}

run_pipeline() {
  local job_id=$1
  shift
  echo "[Pipeline] JOB_ID=${job_id} ($*)"
  env JOB_ID="${job_id}" MCP_SOURCE="${MCP_SOURCE}" "$@" \
    bash datagen/run_openai_pipeline.sh
}

if [[ "${RUN_SINGLE}" == "1" ]]; then
  run_pipeline "stage1_single_${TIMESTAMP}-${MCP_SOURCE}-${SINGLE_INTERACTION_MODE}" \
    MODE=single_server \
    SAMPLING_STRATEGY=uniform \
    SAMPLES_PER_SERVER=5 \
    INTERACTION_MODE="${SINGLE_INTERACTION_MODE}"
fi

if [[ "${RUN_MULTI}" == "1" ]]; then
  for tools in 2 4 6; do
    run_pipeline "stage1_multi${tools}_${TIMESTAMP}-${MCP_SOURCE}-${MULTI_INTERACTION_MODE}" \
      MODE=multi_server \
      NUM_TOOLS="${tools}" \
      TOTAL_PROMPTS=300 \
      SAMPLING_STRATEGY=random \
      INTERACTION_MODE="${MULTI_INTERACTION_MODE}"
  done
fi
