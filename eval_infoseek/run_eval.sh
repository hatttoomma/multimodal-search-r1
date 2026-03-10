#!/usr/bin/env bash
# Standalone multi-turn eval for MMSearch-R1 on InfoSeek.
# Adjust --max-samples for quick debugging (e.g. 5-20).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATA="${DATA:-mmsearch_r1/data/mmsearch_r1_infoseek_sub_2k.parquet}"
OUTPUT="${OUTPUT:-eval_infoseek/results.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"

python eval_infoseek/eval.py \
    --model "$MODEL" \
    --data "$DATA" \
    --output "$OUTPUT" \
    --max-samples "$MAX_SAMPLES" \
    --max-new-tokens 2048 \
    --max-rounds 3 \
    --image-search-limit 1 \
    --text-search-limit 2 \
    --search-penalty 0.1 \
    --format-penalty 0.1 \
    --reward-mode EM \
    --torch-dtype bfloat16
