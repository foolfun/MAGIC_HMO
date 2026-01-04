#!/bin/bash
set -e

# Usage:
# $1 BACKBONE   : qwen | baichuan | glm4 | gpt4o | mistral | gemini
# $2 MODE       : batch | single
# $3 NUMBER     : (batch) number of queries, default -1
# $4 ABLATION   : (batch) "", wo-R, wo-evalR, wo-Imp, wo-Exp, wo-evalGen
# $5 TEST_QUERY : (single) query text

BACKBONE=${1:-qwen}
MODE=${2:-batch}

# =========================
# batch 测试
# =========================
if [ "$MODE" = "batch" ]; then
  NUMBER=${3:--1}
  ABLATION=${4:-""}

  echo "======================"
  echo "Running NAMeGEn in BATCH mode"
  echo "Backbone : $BACKBONE"
  echo "Number   : $NUMBER"
  echo "Ablation : $ABLATION"
  echo "======================"

  python NAMeGEn.py \
    --backbone "$BACKBONE" \
    --mode batch \
    --number "$NUMBER" \
    --ablation "$ABLATION"

# =========================
# single 测试
# =========================
elif [ "$MODE" = "single" ]; then
  TEST_QUERY=${3}

  if [ -z "$TEST_QUERY" ]; then
    echo "Error: TEST_QUERY is required in single mode"
    exit 1
  fi

  echo "======================"
  echo "Running NAMeGEn in SINGLE TEST mode"
  echo "Backbone : $BACKBONE"
  echo "Query    : $TEST_QUERY"
  echo "======================"

  python NAMeGEn.py \
    --backbone "$BACKBONE" \
    --mode single \
    --query "$TEST_QUERY"

else
  echo "Error: MODE must be 'batch' or 'single'"
  exit 1
fi
