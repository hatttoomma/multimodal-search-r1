"""Lightweight test for scoring logic — no GPU/model needed."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mmsearch_r1.utils.reward_score_mm.mmsearch_r1_score import (
    compute_score, extract_solution, em_check, subem_check, format_reward,
)

def test_extract():
    assert extract_solution("<reason>x</reason><answer>Titanic</answer>") == "Titanic"
    assert extract_solution("no answer tag") is None
    print("  extract_solution: OK")

def test_em():
    assert em_check("Rogers Communications", ["Rogers Communications", "Rogers Communications Inc."]) == True
    assert em_check("Apple Inc", ["Rogers Communications"]) == False
    print("  em_check: OK")

def test_format():
    # 1-turn direct answer
    score, cnt = format_reward(["<reason>reason</reason><answer>ans</answer>"])
    assert score == 1 and cnt == 0
    # 2-turn: image search + answer
    score, cnt = format_reward([
        "<reason>reason</reason><search><img></search>",
        "<reason>reason</reason><answer>ans</answer>",
    ])
    assert score == 1 and cnt == 1
    # 2-turn: text search + answer
    score, cnt = format_reward([
        "<reason>reason</reason><text_search>query</text_search>",
        "<reason>reason</reason><answer>ans</answer>",
    ])
    assert score == 1 and cnt == 1
    # 3-turn: image + text + answer
    score, cnt = format_reward([
        "<reason>reason</reason><search><img></search>",
        "<reason>reason</reason><text_search>query</text_search>",
        "<reason>reason</reason><answer>ans</answer>",
    ])
    assert score == 1 and cnt == 2
    print("  format_reward: OK")

def test_compute_score():
    # Correct direct answer
    s = compute_score(
        ["<reason>r</reason><answer>Rogers Communications</answer>"],
        ["Rogers Communications"],
        extra_info={"search_penalty": 0.1, "format_penalty": 0.1, "reward_mode": "EM", "use_search_count_penalty": False},
    )
    assert s == 1.0, f"Expected 1.0, got {s}"

    # Correct with search (2-turn) -> score * (1-search_penalty)
    s = compute_score(
        [
            "<reason>r</reason><search><img></search>",
            "<reason>r</reason><answer>Rogers Communications</answer>",
        ],
        ["Rogers Communications"],
        extra_info={"search_penalty": 0.1, "format_penalty": 0.1, "reward_mode": "EM", "use_search_count_penalty": False},
    )
    expected = (1 - 0.1) * (1 * (1 - 0.1)) + 0.1 * 1  # (1-fp)*score_with_penalty + fp*format
    assert abs(s - expected) < 1e-6, f"Expected {expected}, got {s}"

    # Wrong answer, good format
    s = compute_score(
        ["<reason>r</reason><answer>Apple</answer>"],
        ["Rogers Communications"],
        extra_info={"search_penalty": 0.1, "format_penalty": 0.1, "reward_mode": "EM", "use_search_count_penalty": False},
    )
    assert s == 0.1, f"Expected 0.1 (format only), got {s}"

    # Wrong answer, bad format
    s = compute_score(
        ["some random text without tags"],
        ["Rogers Communications"],
        extra_info={"search_penalty": 0.1, "format_penalty": 0.1, "reward_mode": "EM", "use_search_count_penalty": False},
    )
    assert s == 0.0, f"Expected 0.0, got {s}"

    # Correct threshold check: score > format_penalty + 1e-4
    assert 1.0 > 0.1 + 1e-4  # correct direct answer passes
    assert 0.1 < 0.1 + 1e-4  # wrong answer with format doesn't pass

    print("  compute_score: OK")

if __name__ == "__main__":
    print("Running scoring tests...")
    test_extract()
    test_em()
    test_format()
    test_compute_score()
    print("All tests passed!")
