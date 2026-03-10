import os
from typing import Any, Dict, List

import requests


_SERPER_SEARCH_ENDPOINT = "https://google.serper.dev/search"
_JINA_READER_PREFIX = "https://r.jina.ai/http://"
_DEFAULT_TOP_K = 3
_HTTP_TIMEOUT = 25
_MAX_READER_CHARS = 6000
_DEFAULT_DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _safe_get_top_k() -> int:
    raw_k = os.getenv("MMSEARCH_TEXT_SEARCH_TOP_K", str(_DEFAULT_TOP_K))
    try:
        top_k = int(raw_k)
        return max(top_k, 1)
    except (TypeError, ValueError):
        return _DEFAULT_TOP_K


def _serper_text_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY is not set.")

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": top_k,
    }
    response = requests.post(_SERPER_SEARCH_ENDPOINT, headers=headers, json=payload, timeout=_HTTP_TIMEOUT)
    response.raise_for_status()
    response_data = response.json()
    return (response_data.get("organic") or [])[:top_k]


def _read_with_jina(url: str) -> str:
    cleaned = url.replace("http://", "").replace("https://", "")
    reader_url = _JINA_READER_PREFIX + cleaned
    headers: Dict[str, str] = {}
    jina_api_key = os.getenv("JINA_API_KEY")
    if jina_api_key:
        headers["Authorization"] = f"Bearer {jina_api_key}"
    response = requests.get(reader_url, headers=headers, timeout=_HTTP_TIMEOUT)
    response.raise_for_status()
    text = response.text.strip()
    if len(text) > _MAX_READER_CHARS:
        text = text[:_MAX_READER_CHARS]
    return text


def _resolve_qwen_client_config() -> tuple[str, str, str, Dict[str, str]]:
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    generic_api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    api_key = openrouter_api_key or generic_api_key
    if not api_key:
        raise RuntimeError("QWEN_API_KEY / OPENROUTER_API_KEY / DASHSCOPE_API_KEY is not set.")

    explicit_base = os.getenv("QWEN_API_BASE", "").strip()
    use_openrouter = bool(openrouter_api_key) or "openrouter.ai" in explicit_base
    base_url = explicit_base or (_DEFAULT_OPENROUTER_BASE if use_openrouter else _DEFAULT_DASHSCOPE_BASE)

    model_name = os.getenv("QWEN_SUMMARY_MODEL", "qwen3-32b")
    if use_openrouter and "/" not in model_name:
        # OpenRouter model ids generally require provider prefix, e.g. qwen/qwen3-32b.
        model_name = f"qwen/{model_name}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if use_openrouter:
        # Optional OpenRouter attribution headers.
        if os.getenv("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "")
        if os.getenv("OPENROUTER_X_TITLE"):
            headers["X-Title"] = os.getenv("OPENROUTER_X_TITLE", "")
    return api_key, base_url, model_name, headers


def _summarize_with_qwen(query: str, page_text: str) -> str:
    _, base_url, model_name, headers = _resolve_qwen_client_config()
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "temperature": 0.1,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise web reading assistant. Summarize only information relevant to the query.",
            },
            {
                "role": "user",
                "content": (
                    f"Original query:\n{query}\n\n"
                    "Webpage content:\n"
                    f"{page_text}\n\n"
                    "Please write a concise summary (3-5 sentences), focusing on query-relevant facts only."
                ),
            },
        ],
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=_HTTP_TIMEOUT)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"].strip()
    return content


def _fallback_summary(page_text: str) -> str:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if not lines:
        return "No readable content was extracted from this webpage."
    # Keep lightweight deterministic fallback if Qwen API is unavailable.
    return " ".join(lines[:4])[:450]


def call_text_search(text_query: str):
    """
    Text Search Tool (Serper + JINA Reader + Qwen3-32B).

    Flow:
      1) Serper retrieves top-k relevant webpage links for the text query.
      2) JINA Reader parses and cleans each page.
      3) Qwen3-32B summarizes each page with respect to the original query.

    Returns:
      - tool_returned_str: summarized passages with links.
      - tool_stat: execution status and metadata.
    """
    top_k = _safe_get_top_k()
    tool_success = False
    error_msg = ""
    summarized_count = 0

    tool_returned_str = (
        "[Text Search Results] Below are the text summaries of the most relevant webpages "
        "related to your query, ranked in descending order of relevance:\n"
    )

    try:
        search_results = _serper_text_search(query=text_query, top_k=top_k)
        for idx, result in enumerate(search_results, start=1):
            link = (result.get("link") or "").strip()
            if not link:
                continue

            page_text = _read_with_jina(link)
            try:
                summary = _summarize_with_qwen(query=text_query, page_text=page_text)
            except Exception:
                summary = _fallback_summary(page_text)

            tool_returned_str += f"{idx}. ({link}) {summary}\n"
            summarized_count += 1

        tool_success = summarized_count > 0
        if not tool_success:
            error_msg = "No valid search results were summarized."
    except Exception as exc:
        error_msg = str(exc)

    if not tool_success:
        tool_returned_str = (
            "[Text Search Results] There is an error encountered in performing search. "
            "Please reason with your own capaibilities."
        )

    tool_stat = {
        "success": tool_success,
        "num_results": summarized_count,
        "top_k": top_k,
        "error": error_msg,
    }
    return tool_returned_str, tool_stat