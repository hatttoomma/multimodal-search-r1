import os
from io import BytesIO
from typing import Any, Dict, List

import requests
from PIL import Image


_SERPER_LENS_ENDPOINT = "https://google.serper.dev/lens"
_DEFAULT_TOP_K = 3
_HTTP_TIMEOUT = 20


def _safe_get_top_k() -> int:
    raw_k = os.getenv("MMSEARCH_IMAGE_SEARCH_TOP_K", str(_DEFAULT_TOP_K))
    try:
        top_k = int(raw_k)
        return max(top_k, 1)
    except (TypeError, ValueError):
        return _DEFAULT_TOP_K


def _download_image(image_url: str) -> Image.Image | None:
    if not image_url:
        return None
    try:
        response = requests.get(image_url, timeout=_HTTP_TIMEOUT)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except Exception:
        return None


def _serper_lens_search(image_url: str, top_k: int) -> List[Dict[str, Any]]:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY is not set.")

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "url": image_url,
    }
    response = requests.post(_SERPER_LENS_ENDPOINT, headers=headers, json=payload, timeout=_HTTP_TIMEOUT)
    response.raise_for_status()
    response_data = response.json()

    # Serper lens payload shapes may vary; keep broad field fallbacks.
    candidates = (
        response_data.get("visualMatches")
        or response_data.get("visual_matches")
        or response_data.get("organic")
        or response_data.get("related_content")
        or []
    )
    return candidates[:top_k]


def call_image_search(image_url: str):
    """
    Image Search Tool (Serper only).

    Input:
      - image_url: query image URL.
    Flow:
      - Use Serper Google Lens endpoint to retrieve visually relevant web pages.
      - Extract each result's title and thumbnail.
    Output:
      - Interleaved thumbnail placeholders + titles as text.
      - Actual PIL thumbnail images in the same order.
      - Tool stats.
    """
    top_k = _safe_get_top_k()
    tool_returned_images: List[Image.Image] = []
    tool_returned_str = (
        "[Image Search Results] The result of the image search consists of web page information "
        "related to the image from the user's original question. Each result includes the main "
        "image from the web page and its title, ranked in descending order of search relevance, "
        "as demonstrated below:\n"
    )

    tool_success = False
    error_msg = ""
    used_results = 0

    try:
        results = _serper_lens_search(image_url=image_url, top_k=top_k)
        for idx, item in enumerate(results, start=1):
            title = (
                item.get("title")
                or item.get("source")
                or item.get("domain")
                or "Untitled result"
            ).strip()
            thumb_url = (
                item.get("thumbnail")
                or item.get("image")
                or item.get("imageUrl")
                or ""
            )
            thumb_image = _download_image(thumb_url)
            if thumb_image is None:
                # Skip entries without fetchable thumbnails to keep text/image alignment.
                continue

            tool_returned_images.append(thumb_image)
            tool_returned_str += (
                f"{idx}. image: <|vision_start|><|image_pad|><|vision_end|>\n"
                f"title: {title}\n"
            )
            used_results += 1

        tool_success = used_results > 0
        if not tool_success:
            error_msg = "No valid image results with downloadable thumbnails."
    except Exception as exc:
        error_msg = str(exc)

    if not tool_success:
        tool_returned_str = (
            "[Image Search Results] There is an error encountered in performing search. "
            "Please reason with your own capaibilities."
        )
        tool_returned_images = []

    tool_stat = {
        "success": tool_success,
        "num_images": len(tool_returned_images),
        "top_k": top_k,
        "error": error_msg,
    }
    return tool_returned_str, tool_returned_images, tool_stat