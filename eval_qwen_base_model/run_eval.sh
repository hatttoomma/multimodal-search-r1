#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

conda run --no-capture-output -n mmsearch_r1 \
    python3 eval_qwen_base_model/eval.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --data mmsearch_r1/data/mmsearch_r1_infoseek_sub_2k.parquet \
    --output eval_qwen_base_model/results.jsonl \
    --max-tokens 256 \
    --temperature 0.0 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --batch-size 64 \
    "$@"
