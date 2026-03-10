"""
Standalone multi-turn evaluation for MMSearch-R1 on InfoSeek.

Replicates the training-time validation logic (multi-turn generation with
image/text search tool calls) using plain HuggingFace generate() — no Ray,
no vLLM.  Useful for debugging and small-scale experiments.

Usage:
    python eval_infoseek/eval.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --data mmsearch_r1/data/mmsearch_r1_infoseek_sub_2k.parquet \
        --max-samples 20
"""

import argparse
import json
import math
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import List, Optional

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from mmsearch_r1.utils.tools.image_search import call_image_search
from mmsearch_r1.utils.tools.text_search import call_text_search
from mmsearch_r1.utils.reward_score_mm.mmsearch_r1_score import (
    compute_score as mmsearch_r1_compute_score,
    extract_solution,
)

# ---------------------------------------------------------------------------
# Prompt templates (same as training)
# ---------------------------------------------------------------------------
_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "mmsearch_r1", "prompts")


def _load_prompt_pkl(filename: str) -> str:
    path = os.path.join(_PROMPT_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


ROUND1_USER_PROMPT = _load_prompt_pkl("round_1_user_prompt_qwenvl.pkl")
AFTER_IMAGE_SEARCH_PROMPT = _load_prompt_pkl("after_image_search_prompt_qwenvl.pkl")
AFTER_TEXT_SEARCH_PROMPT = _load_prompt_pkl("after_text_search_prompt_qwenvl.pkl")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(parquet_path: str, max_samples: Optional[int] = None):
    df = pd.read_parquet(parquet_path)
    if max_samples is not None:
        df = df.head(max_samples)

    samples = []
    for _, row in df.iterrows():
        question = row["prompt"][0]["content"]
        img_bytes = row["images"][0]["bytes"]
        image = Image.open(BytesIO(img_bytes)).convert("RGB")

        reward_info = row["reward_model"]
        gt = reward_info["ground_truth"]
        ground_truth_list = [gt] if isinstance(gt, str) else list(gt)
        if "candidate_answers" in reward_info and reward_info["candidate_answers"]:
            cands = reward_info["candidate_answers"]
            if isinstance(cands, list):
                ground_truth_list += cands
            elif isinstance(cands, str):
                ground_truth_list += json.loads(cands)
        ground_truth_list = [g for g in ground_truth_list if isinstance(g, str)]

        samples.append({
            "data_id": row["data_id"],
            "question": question,
            "image": image,
            "ground_truth": gt,
            "candidate_answers": ground_truth_list,
        })
    return samples


# ---------------------------------------------------------------------------
# Image processing helper (mirrors mm_rl_dataset.process_image)
# ---------------------------------------------------------------------------


def process_image(image: Image.Image, max_pixels=672 * 672 * 2, min_pixels=512 * 512) -> Image.Image:
    if (image.width * image.height) > max_pixels:
        factor = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize(
            (int(image.width * factor), int(image.height * factor)),
            resample=Image.Resampling.NEAREST,
        )
    if (image.width * image.height) < min_pixels:
        factor = math.sqrt(min_pixels / (image.width * image.height))
        image = image.resize(
            (int(image.width * factor), int(image.height * factor)),
            resample=Image.Resampling.NEAREST,
        )
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


# ---------------------------------------------------------------------------
# Multi-turn generation for a single sample
# ---------------------------------------------------------------------------


def generate_multi_turn(
    model,
    processor,
    sample: dict,
    max_new_tokens: int = 2048,
    max_rounds: int = 3,
    image_search_limit: int = 1,
    text_search_limit: int = 2,
    parallel_search: bool = True,
    device: str = "cuda",
) -> dict:
    """
    Runs the multi-turn generation loop for one sample, mirroring the
    vLLMRollout_MultiTurn_MMSearch_R1 logic:

      Round 0: model sees user prompt + image -> generates response
        - If response ends with <search><img></search> -> call image search
        - If response ends with <text_search>...</text_search> -> call text search
        - Otherwise -> done (direct answer)
      Round 1+: inject search results as new user turn -> generate again
        - repeat until max_rounds or model gives a direct answer
    """
    question = sample["question"]
    user_image = process_image(sample["image"])

    # Build round-1 user message (same as training: ROUND1_USER_PROMPT + question)
    round1_text = ROUND1_USER_PROMPT + question

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": user_image},
                {"type": "text", "text": round1_text},
            ],
        }
    ]

    all_images = [user_image]
    all_response_texts = []  # list of assistant response strings per round
    image_search_cnt = 0
    text_search_cnt = 0

    for round_idx in range(max_rounds):
        # Tokenize conversation so far
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_prompt],
            images=all_images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        response_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        all_response_texts.append(response_text)

        # Append assistant response to messages
        messages.append({"role": "assistant", "content": response_text})

        # Check if model wants to call a tool
        needs_image_search = bool(re.search(r"<search><img></search>$", response_text.strip()))
        text_search_match = re.search(r"<text_search>(.*?)</text_search>$", response_text.strip(), re.DOTALL)
        needs_text_search = bool(text_search_match)

        if needs_image_search and image_search_cnt < image_search_limit and round_idx < max_rounds - 1:
            # --- Call image search ---
            image_search_cnt += 1
            search_result_str, search_result_imgs, tool_stat = call_image_search(
                image_url=None, image=user_image
            )

            # Build next user turn
            if "[Image Search Results]" in search_result_str and "error" not in search_result_str.lower():
                org_query = question
                search_message = (
                    "Searched results: <information>"
                    + search_result_str
                    + "</information>\n"
                    + f"Original user's question: {org_query}\n"
                    + AFTER_IMAGE_SEARCH_PROMPT
                )
            else:
                search_message = search_result_str

            # Add search result images
            user_content = []
            if search_result_imgs:
                for img in search_result_imgs:
                    all_images.append(img)
                    user_content.append({"type": "image", "image": img})
            user_content.append({"type": "text", "text": search_message})

            messages.append({"role": "user", "content": user_content})

        elif needs_text_search and text_search_cnt < text_search_limit and round_idx < max_rounds - 1:
            # --- Call text search ---
            text_search_cnt += 1
            text_query = text_search_match.group(1).strip()
            search_result_str, tool_stat = call_text_search(text_query=text_query)

            if "[Text Search Results]" in search_result_str and "error" not in search_result_str.lower():
                org_query = question
                search_message = (
                    "Searched results: <information>"
                    + search_result_str
                    + "</information>\n"
                    + f"Original user's question: {org_query}\n"
                    + AFTER_TEXT_SEARCH_PROMPT
                )
            else:
                search_message = search_result_str

            messages.append({"role": "user", "content": search_message})

        else:
            # Direct answer or limits reached
            break

    return {
        "data_id": sample["data_id"],
        "question": question,
        "ground_truth": sample["ground_truth"],
        "candidate_answers": sample["candidate_answers"],
        "all_responses": all_response_texts,
        "num_rounds": len(all_response_texts),
        "image_search_cnt": image_search_cnt,
        "text_search_cnt": text_search_cnt,
    }


# ---------------------------------------------------------------------------
# Scoring (identical to training-time reward)
# ---------------------------------------------------------------------------


def score_result(result: dict, search_penalty=0.1, format_penalty=0.1, reward_mode="EM") -> dict:
    """Score a single result using the same logic as training-time reward."""
    extra_info = {
        "search_penalty": search_penalty,
        "format_penalty": format_penalty,
        "reward_mode": reward_mode,
        "use_search_count_penalty": False,
    }
    score = mmsearch_r1_compute_score(
        prediction=result["all_responses"],
        ground_truth=result["candidate_answers"],
        extra_info=extra_info,
    )

    answer = extract_solution(result["all_responses"][-1]) if result["all_responses"] else None
    correct = score > (format_penalty + 1e-4)

    result["extracted_answer"] = answer
    result["score"] = float(score)
    result["correct"] = correct
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Standalone multi-turn eval for MMSearch-R1 on InfoSeek")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data", type=str, default="mmsearch_r1/data/mmsearch_r1_infoseek_sub_2k.parquet")
    parser.add_argument("--output", type=str, default="eval_infoseek/results.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-rounds", type=int, default=3, help="Max multi-turn rounds")
    parser.add_argument("--image-search-limit", type=int, default=1)
    parser.add_argument("--text-search-limit", type=int, default=2)
    parser.add_argument("--search-penalty", type=float, default=0.1)
    parser.add_argument("--format-penalty", type=float, default=0.1)
    parser.add_argument("--reward-mode", type=str, default="EM", choices=["EM", "SubEM"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.torch_dtype]

    print(f"Loading model: {args.model}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        device_map=args.device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)

    print(f"Loading data: {args.data}")
    samples = load_data(args.data, max_samples=args.max_samples)
    print(f"Total samples: {len(samples)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    correct_cnt = 0
    total = 0
    search_text_cnt = 0
    search_image_cnt = 0
    search_mix_cnt = 0
    t0 = time.time()

    with open(args.output, "w") as out_f:
        for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
            result = generate_multi_turn(
                model=model,
                processor=processor,
                sample=sample,
                max_new_tokens=args.max_new_tokens,
                max_rounds=args.max_rounds,
                image_search_limit=args.image_search_limit,
                text_search_limit=args.text_search_limit,
                device=args.device,
            )

            result = score_result(
                result,
                search_penalty=args.search_penalty,
                format_penalty=args.format_penalty,
                reward_mode=args.reward_mode,
            )

            total += 1
            if result["correct"]:
                correct_cnt += 1

            if result["image_search_cnt"] > 0 and result["text_search_cnt"] == 0:
                search_image_cnt += 1
            elif result["text_search_cnt"] > 0 and result["image_search_cnt"] == 0:
                search_text_cnt += 1
            elif result["image_search_cnt"] > 0 and result["text_search_cnt"] > 0:
                search_mix_cnt += 1

            # Write result (drop PIL image)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                elapsed = time.time() - t0
                acc = correct_cnt / total if total > 0 else 0
                print(
                    f"  [{total}/{len(samples)}] "
                    f"Acc: {correct_cnt}/{total} = {acc:.4f} | "
                    f"Search(img/text/mix): {search_image_cnt}/{search_text_cnt}/{search_mix_cnt} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

    elapsed = time.time() - t0
    acc = correct_cnt / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"Final Accuracy: {correct_cnt}/{total} = {acc:.4f}")
    print(f"Search ratios — image_only: {search_image_cnt/total:.3f}, "
          f"text_only: {search_text_cnt/total:.3f}, "
          f"mix: {search_mix_cnt/total:.3f}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
