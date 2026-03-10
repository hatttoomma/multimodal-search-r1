import os
from io import BytesIO
from typing import Any, Dict, List, Optional

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


def _pil_image_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def _serper_lens_search_by_url(image_url: str, api_key: str, top_k: int) -> List[Dict[str, Any]]:
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"url": image_url}
    response = requests.post(
        _SERPER_LENS_ENDPOINT, headers=headers, json=payload, timeout=_HTTP_TIMEOUT
    )
    response.raise_for_status()
    return _extract_candidates(response.json(), top_k)


def _serper_lens_search_by_image(image: Image.Image, api_key: str, top_k: int) -> List[Dict[str, Any]]:
    headers = {"X-API-KEY": api_key}
    image_bytes = _pil_image_to_bytes(image)
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    response = requests.post(
        _SERPER_LENS_ENDPOINT, headers=headers, files=files, timeout=_HTTP_TIMEOUT
    )
    response.raise_for_status()
    return _extract_candidates(response.json(), top_k)


def _extract_candidates(response_data: dict, top_k: int) -> List[Dict[str, Any]]:
    candidates = (
        response_data.get("visualMatches")
        or response_data.get("visual_matches")
        or response_data.get("organic")
        or response_data.get("related_content")
        or []
    )
    return candidates[:top_k]


def _serper_lens_search(
    top_k: int,
    image_url: Optional[str] = None,
    image: Optional[Image.Image] = None,
) -> List[Dict[str, Any]]:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY is not set.")

    if image_url:
        return _serper_lens_search_by_url(image_url, api_key, top_k)
    elif image is not None:
        return _serper_lens_search_by_image(image, api_key, top_k)
    else:
        raise ValueError("Either image_url or image must be provided.")


def call_image_search(
    image_url: Optional[str] = None,
    image: Optional[Image.Image] = None,
) -> tuple:
    """
    Image Search Tool (Serper only).

    Accepts either *image_url* (str) or a PIL *image*.  When a URL is
    available it is preferred; when the URL is ``None`` the PIL image is
    uploaded directly via multipart form-data.

    Returns:
      (tool_returned_str, tool_returned_images, tool_stat)
    """
    if not image_url and image is None:
        return (
            "[Image Search Results] There is an error encountered in performing search. "
            "Please reason with your own capaibilities.",
            [],
            {"success": False, "num_images": 0, "top_k": 0, "error": "No image_url or image provided."},
        )

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
        results = _serper_lens_search(top_k=top_k, image_url=image_url, image=image)
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