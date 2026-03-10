"""
Evaluate Qwen2.5-VL-3B-Instruct on mmsearch_r1_infoseek_sub_2k using vLLM offline inference.
"""

import argparse
import base64
import io
import json
import os
import time

import pyarrow.parquet as pq
from PIL import Image
from vllm import LLM, SamplingParams


def load_data(parquet_path: str):
    """Load parquet row-by-row, yielding dicts with question, image bytes, ground truth."""
    pf = pq.ParquetFile(parquet_path)
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx)
        df = table.to_pandas()
        for _, row in df.iterrows():
            question = row["prompt"][0]["content"]
            img_bytes = row["images"][0]["bytes"]

            reward_model_info = row["reward_model"]
            gt = reward_model_info["ground_truth"]
            ground_truth_list = [gt] if isinstance(gt, str) else list(gt)

            if "candidate_answers" in reward_model_info and reward_model_info["candidate_answers"]:
                candidates_raw = reward_model_info["candidate_answers"]
                if isinstance(candidates_raw, list):
                    ground_truth_list += candidates_raw
                elif isinstance(candidates_raw, str):
                    ground_truth_list += json.loads(candidates_raw)

            ground_truth_list = [g for g in ground_truth_list if isinstance(g, str)]

            yield {
                "data_id": row["data_id"],
                "question": question,
                "image_bytes": img_bytes,
                "ground_truth": gt,
                "candidate_answers": ground_truth_list,
            }


def image_bytes_to_data_url(img_bytes: bytes) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_messages(question: str, image_data_url: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": question},
            ],
        }
    ]


def extract_numbers(text: str) -> list[float]:
    """Extract all numbers (int or float) from text."""
    import re
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text)]


def check_answer(response: str, candidate_answers: list) -> bool:
    response_lower = response.strip().lower()
    for ans in candidate_answers:
        if isinstance(ans, str):
            if ans.lower() in response_lower:
                return True
        elif isinstance(ans, dict) and "range" in ans:
            low, high = ans["range"]
            for num in extract_numbers(response):
                if low <= num <= high:
                    return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--data",
        type=str,
        default="mmsearch_r1/data/mmsearch_r1_infoseek_sub_2k.parquet",
    )
    parser.add_argument("--output", type=str, default="eval_qwen_base_model/results.jsonl")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for debugging")
    parser.add_argument("--batch-size", type=int, default=64, help="Requests per vLLM batch")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(f"Loading data from: {args.data}")
    all_samples = list(load_data(args.data))
    if args.max_samples:
        all_samples = all_samples[: args.max_samples]
    print(f"Total samples: {len(all_samples)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_f = open(args.output, "w")
    correct = 0
    total = 0
    t0 = time.time()

    for batch_start in range(0, len(all_samples), args.batch_size):
        batch = all_samples[batch_start : batch_start + args.batch_size]

        conversations = []
        for sample in batch:
            data_url = image_bytes_to_data_url(sample["image_bytes"])
            conversations.append(build_messages(sample["question"], data_url))

        outputs = llm.chat(
            messages=conversations,
            sampling_params=sampling_params,
        )

        for sample, output in zip(batch, outputs):
            response_text = output.outputs[0].text
            is_correct = check_answer(response_text, sample["candidate_answers"])
            correct += int(is_correct)
            total += 1

            result = {
                "data_id": sample["data_id"],
                "question": sample["question"],
                "ground_truth": sample["ground_truth"],
                "candidate_answers": sample["candidate_answers"],
                "response": response_text,
                "correct": is_correct,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

        out_f.flush()
        elapsed = time.time() - t0
        print(
            f"  [{total}/{len(all_samples)}] "
            f"Accuracy: {correct}/{total} = {correct / total:.4f} | "
            f"Elapsed: {elapsed:.1f}s"
        )

    out_f.close()
    elapsed = time.time() - t0
    print(f"\nDone! Final accuracy: {correct}/{total} = {correct / total:.4f}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
