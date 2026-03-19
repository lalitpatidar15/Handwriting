import base64
import asyncio
import os
import json
import re
import time
import threading
from typing import Any, Optional


def _extract_retry_delay(err_str: str) -> int:
    """Parse retry_delay seconds from a Gemini quota-exceeded error string."""
    m = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', err_str)
    return int(m.group(1)) if m else 0

import requests


def _strip_json_fences(text):
    """Remove markdown code fences from model output before JSON parsing."""
    if not text:
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def cloud_ocr_gemini(image_path):
    """
    Google Gemini Vision OCR via google-genai SDK.
    Model fallback chain skips per-day quota exhaustion and honours
    per-minute retry_delay (up to 30 s) before giving up on a model.
    """
    from google import genai as gai
    import PIL.Image

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Get free key from: https://aistudio.google.com/app/apikey")

    client = gai.Client(api_key=api_key)

    # gemini-2.0-flash has limit:0 on free tier; gemini-1.5-flash-8b is the
    # most generous free-tier fallback available on the new SDK.
    model_names = [
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ]

    img = PIL.Image.open(image_path)
    prompt = (
        "You are an Enterprise Intelligent Document Processing (IDP) AI. "
        "Analyze this document (handwritten note, form, bill, or prescription) and extract the data strictly as a JSON object. "
        "Rules: "
        "1. Do NOT wrap JSON in markdown code blocks. Just return raw JSON. "
        "2. Do NOT add any conversational text. "
        "3. Use this exact schema: "
        '{"document_type": "string", "confidence_score": 0.95, '
        '"metadata": {"date": null, "patient_or_client_name": null}, '
        '"extracted_data": [], "full_raw_text": "string"}'
    )

    last_error = None
    for model_name in model_names:
        print(f"[OCR] Trying Gemini model: {model_name}")
        try:
            for attempt in range(2):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[prompt, img],
                    )
                    text_response = _strip_json_fences((response.text or "").strip())
                    try:
                        return json.loads(text_response)
                    except json.JSONDecodeError:
                        print(f"[OCR] Gemini {model_name} returned non-JSON, using raw text.")
                        return {"full_raw_text": text_response, "document_type": "UNKNOWN"}
                except Exception as inner_exc:
                    err = str(inner_exc)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        # Per-day quota: skip this model immediately
                        if "PerDay" in err or "PerDayPer" in err:
                            raise
                        # Per-minute quota: wait if delay is short, then retry once
                        delay = _extract_retry_delay(err)
                        if attempt == 0 and 0 < delay <= 30:
                            print(f"[OCR] {model_name} rate-limited, waiting {delay}s…")
                            time.sleep(delay)
                            continue
                    raise
        except Exception as e:
            last_error = e
            print(f"[OCR] Gemini {model_name} failed: {e}")
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini OCR failed for all model variants")


def cloud_ocr_gemini_structured(image_path):
    """Extract structured form content using Gemini (google-genai SDK) and return canonical JSON dict."""
    from google import genai as gai
    import PIL.Image

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Get free key from: https://aistudio.google.com/app/apikey")

    client = gai.Client(api_key=api_key)

    model_names = [
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ]
    last_error = None

    img = PIL.Image.open(image_path)
    prompt = (
        "You are a form extraction engine. Read this form image and return ONLY valid JSON with this schema: "
        '{"document_type": string, "full_text": string, "fields": [{"label": string, "value": string}], "table": [object]}. '
        "Rules: 1) Preserve field labels exactly. 2) Put handwritten or filled values in value. "
        "3) If value is empty, use empty string. 4) Do not include markdown or explanations."
    )

    for model_name in model_names:
        print(f"[OCR] Trying Gemini structured model: {model_name}")
        try:
            for attempt in range(2):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[prompt, img],
                    )
                    raw = _strip_json_fences((response.text or "").strip())
                    parsed = json.loads(raw)

                    fields = parsed.get("fields", []) or []
                    normalized_fields = [
                        {
                            "label": str(item.get("label", "") or "").strip(),
                            "value": str(item.get("value", "") or "").strip(),
                        }
                        for item in fields
                        if isinstance(item, dict)
                    ]

                    table = parsed.get("table", [])
                    if not isinstance(table, list):
                        table = []

                    return {
                        "document_type": str(parsed.get("document_type", "STRUCTURED_FORM") or "STRUCTURED_FORM"),
                        "full_text": str(parsed.get("full_text", "") or ""),
                        "fields": normalized_fields,
                        "table": table,
                    }
                except Exception as inner_exc:
                    err = str(inner_exc)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        if "PerDay" in err or "PerDayPer" in err:
                            raise
                        delay = _extract_retry_delay(err)
                        if attempt == 0 and 0 < delay <= 30:
                            print(f"[OCR] {model_name} rate-limited, waiting {delay}s…")
                            time.sleep(delay)
                            continue
                    raise
        except Exception as e:
            last_error = e
            print(f"[OCR] Gemini structured {model_name} failed: {e}")
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini structured OCR failed for all model variants")

def cloud_ocr_openai(image_path):
    """
    OpenAI GPT-4o Vision Path (Paid).
    Provides human-level transcription for complex handwriting.
    """
    from openai import OpenAI  # type: ignore[attr-defined]
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")

    client = OpenAI(api_key=api_key)
    base64_image = encode_image(image_path)

    prompt = """
    You are an Enterprise Intelligent Document Processing (IDP) AI. 
    Analyze this document and extract the data strictly as a JSON object.
    
    Rules for output:
    1. Do NOT wrap the JSON in markdown code blocks like ```json ... ```. Just return raw JSON.
    2. Do NOT add any conversational text.
    3. Use this exact schema format:
    {
      "document_type": "string",
      "confidence_score": 0.95,
      "metadata": {},
      "extracted_data": [],
      "full_raw_text": "string containing all the text"
    }
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=1500,
    )

    text_response = response.choices[0].message.content.strip()
    
    # Clean up potential markdown formatting
    if text_response.startswith('```json'): text_response = text_response[7:]
    elif text_response.startswith('```'): text_response = text_response[3:]
    if text_response.endswith('```'): text_response = text_response[:-3]
    
    try:
        return json.loads(text_response.strip())
    except json.JSONDecodeError:
        return {"full_raw_text": text_response.strip(), "document_type": "UNKNOWN"}

def cloud_ocr(image_path, provider="gemini"):
    """
    Universal Cloud OCR - Supports both Gemini (FREE) and OpenAI (Paid).
    Default: Gemini (free tier)
    """
    if provider.lower() == "gemini":
        return cloud_ocr_gemini(image_path)
    elif provider.lower() == "openai":
        return cloud_ocr_openai(image_path)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'gemini' or 'openai'")


def cloud_ocr_structured(image_path, provider="gemini"):
    """Structured OCR endpoint; currently Gemini is supported for form JSON extraction."""
    if provider.lower() == "gemini":
        return cloud_ocr_gemini_structured(image_path)
    raise ValueError(f"Structured OCR currently supports provider='gemini' only, got: {provider}")


def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_bbox(vertices: list[dict[str, Any]]) -> list[float]:
    xs = [float(v.get("x", 0.0) or 0.0) for v in vertices]
    ys = [float(v.get("y", 0.0) or 0.0) for v in vertices]
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0]
    return [min(xs), min(ys), max(xs), max(ys)]


def _normalize_tokens_to_legacy_words(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    legacy = []
    for tok in tokens:
        bbox = tok.get("bbox", [0.0, 0.0, 1.0, 1.0])
        if len(bbox) != 4:
            bbox = [0.0, 0.0, 1.0, 1.0]
        legacy.append(
            {
                "text": str(tok.get("text", "") or ""),
                "confidence": _safe_float(tok.get("confidence", 0.0), 0.0),
                "box": ((bbox[0], bbox[1]), (bbox[2], bbox[3])),
                "bbox": bbox,
                "source_engine": str(tok.get("source_engine", "specialized_ocr") or "specialized_ocr"),
                "page": int(tok.get("page", 1) or 1),
            }
        )
    return legacy


def _poll_azure_result(operation_url: str, headers: dict[str, str], timeout_sec: int = 30) -> dict[str, Any]:
    started = time.time()
    while time.time() - started < timeout_sec:
        response = requests.get(operation_url, headers=headers, timeout=20)
        response.raise_for_status()
        payload = response.json()
        status = str(payload.get("status", "")).lower()
        if status in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(1.2)
    raise TimeoutError("Azure Document Intelligence analysis timed out")


def ocr_layout_google_document_ai(image_path: str) -> dict[str, Any]:
    project_id = os.getenv("GOOGLE_DOCUMENT_AI_PROJECT_ID")
    location = os.getenv("GOOGLE_DOCUMENT_AI_LOCATION", "us")
    processor_id = os.getenv("GOOGLE_DOCUMENT_AI_PROCESSOR_ID")
    api_key = os.getenv("GOOGLE_DOCUMENT_AI_API_KEY")

    if not (project_id and processor_id and api_key):
        raise ValueError(
            "Missing Document AI env vars. Required: GOOGLE_DOCUMENT_AI_PROJECT_ID, GOOGLE_DOCUMENT_AI_PROCESSOR_ID, GOOGLE_DOCUMENT_AI_API_KEY"
        )

    mime_type = "application/pdf" if image_path.lower().endswith(".pdf") else "image/jpeg"
    endpoint = (
        f"https://{location}-documentai.googleapis.com/v1/projects/{project_id}/"
        f"locations/{location}/processors/{processor_id}:process?key={api_key}"
    )
    payload = {
        "rawDocument": {
            "content": _image_to_base64(image_path),
            "mimeType": mime_type,
        }
    }
    response = requests.post(endpoint, json=payload, timeout=40)
    response.raise_for_status()
    data = response.json().get("document", {})

    pages = data.get("pages", []) or []
    tokens = []
    blocks = []
    for page_index, page in enumerate(pages, start=1):
        for tok in page.get("tokens", []) or []:
            layout = tok.get("layout", {}) or {}
            text_anchor = layout.get("textAnchor", {}) or {}
            text_segments = text_anchor.get("textSegments", []) or []
            token_text = ""
            if text_segments:
                start_idx = int(text_segments[0].get("startIndex", 0) or 0)
                end_idx = int(text_segments[0].get("endIndex", 0) or 0)
                token_text = str(data.get("text", ""))[start_idx:end_idx]
            bbox = _normalize_bbox((layout.get("boundingPoly", {}) or {}).get("vertices", []) or [])
            tokens.append(
                {
                    "text": token_text.strip(),
                    "confidence": _safe_float(layout.get("confidence", 0.0), 0.0),
                    "bbox": bbox,
                    "page": page_index,
                    "source_engine": "google_document_ai",
                }
            )

        for paragraph in page.get("paragraphs", []) or []:
            layout = paragraph.get("layout", {}) or {}
            blocks.append(
                {
                    "type": "paragraph",
                    "bbox": _normalize_bbox((layout.get("boundingPoly", {}) or {}).get("vertices", []) or []),
                    "confidence": _safe_float(layout.get("confidence", 0.0), 0.0),
                    "page": page_index,
                    "source_engine": "google_document_ai",
                }
            )

    return {
        "provider": "documentai",
        "full_text": str(data.get("text", "") or ""),
        "tokens": [t for t in tokens if t.get("text")],
        "layout_blocks": blocks,
        "raw": data,
    }


def ocr_layout_aws_textract(image_path: str) -> dict[str, Any]:
    try:
        import boto3
    except Exception as exc:
        raise ImportError("boto3 is required for AWS Textract support: pip install boto3") from exc

    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION", "us-east-1")
    if not (aws_key and aws_secret):
        raise ValueError("Missing AWS credentials. Required: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")

    client = boto3.client(
        "textract",
        region_name=region,
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
    )
    with open(image_path, "rb") as fh:
        raw = fh.read()

    response = client.analyze_document(
        Document={"Bytes": raw},
        FeatureTypes=["FORMS", "TABLES"],
    )
    blocks_raw = response.get("Blocks", []) or []
    words = [b for b in blocks_raw if b.get("BlockType") == "WORD"]

    tokens = []
    layout_blocks = []
    for word in words:
        box = ((word.get("Geometry", {}) or {}).get("BoundingBox", {}) or {})
        left = _safe_float(box.get("Left", 0.0), 0.0)
        top = _safe_float(box.get("Top", 0.0), 0.0)
        width = _safe_float(box.get("Width", 0.0), 0.0)
        height = _safe_float(box.get("Height", 0.0), 0.0)
        tokens.append(
            {
                "text": str(word.get("Text", "") or ""),
                "confidence": _safe_float(word.get("Confidence", 0.0), 0.0) / 100.0,
                "bbox": [left, top, left + width, top + height],
                "page": int(word.get("Page", 1) or 1),
                "source_engine": "aws_textract",
            }
        )

    for item in blocks_raw:
        btype = str(item.get("BlockType", "") or "").lower()
        if btype not in {"table", "cell", "line", "key_value_set"}:
            continue
        box = ((item.get("Geometry", {}) or {}).get("BoundingBox", {}) or {})
        left = _safe_float(box.get("Left", 0.0), 0.0)
        top = _safe_float(box.get("Top", 0.0), 0.0)
        width = _safe_float(box.get("Width", 0.0), 0.0)
        height = _safe_float(box.get("Height", 0.0), 0.0)
        layout_blocks.append(
            {
                "type": btype,
                "bbox": [left, top, left + width, top + height],
                "confidence": _safe_float(item.get("Confidence", 0.0), 0.0) / 100.0,
                "page": int(item.get("Page", 1) or 1),
                "source_engine": "aws_textract",
            }
        )

    return {
        "provider": "textract",
        "full_text": " ".join(t["text"] for t in tokens if t["text"]),
        "tokens": tokens,
        "layout_blocks": layout_blocks,
        "raw": response,
    }


def ocr_layout_azure_docint(image_path: str) -> dict[str, Any]:
    endpoint = os.getenv("AZURE_DOCINT_ENDPOINT")
    api_key = os.getenv("AZURE_DOCINT_API_KEY")
    api_version = os.getenv("AZURE_DOCINT_API_VERSION", "2024-07-31-preview")

    if not (endpoint and api_key):
        raise ValueError("Missing Azure credentials. Required: AZURE_DOCINT_ENDPOINT and AZURE_DOCINT_API_KEY")

    analyze_url = (
        f"{endpoint.rstrip('/')}/documentintelligence/documentModels/prebuilt-layout:analyze"
        f"?api-version={api_version}"
    )
    with open(image_path, "rb") as fh:
        raw = fh.read()

    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/octet-stream",
    }
    response = requests.post(analyze_url, headers=headers, data=raw, timeout=30)
    response.raise_for_status()
    operation_url = response.headers.get("operation-location")
    if not operation_url:
        raise RuntimeError("Azure analyze call did not return operation-location header")

    payload = _poll_azure_result(
        operation_url,
        headers={"Ocp-Apim-Subscription-Key": api_key},
    )
    result = (payload.get("analyzeResult", {}) or {})

    tokens = []
    layout_blocks = []
    for page in result.get("pages", []) or []:
        page_number = int(page.get("pageNumber", 1) or 1)
        for word in page.get("words", []) or []:
            polygon = word.get("polygon", []) or []
            xs = polygon[0::2] if polygon else []
            ys = polygon[1::2] if polygon else []
            bbox = [min(xs or [0.0]), min(ys or [0.0]), max(xs or [0.0]), max(ys or [0.0])]
            tokens.append(
                {
                    "text": str(word.get("content", "") or ""),
                    "confidence": _safe_float(word.get("confidence", 0.0), 0.0),
                    "bbox": bbox,
                    "page": page_number,
                    "source_engine": "azure_docint",
                }
            )

    for table in result.get("tables", []) or []:
        regions = table.get("boundingRegions", []) or []
        poly = []
        if regions:
            poly = (regions[0] or {}).get("polygon", []) or []
        xs = poly[0::2] if poly else []
        ys = poly[1::2] if poly else []
        layout_blocks.append(
            {
                "type": "table",
                "bbox": [min(xs or [0.0]), min(ys or [0.0]), max(xs or [0.0]), max(ys or [0.0])],
                "confidence": 0.9,
                "page": int((regions[0] or {}).get("pageNumber", 1) or 1) if regions else 1,
                "source_engine": "azure_docint",
            }
        )

    return {
        "provider": "azure",
        "full_text": " ".join(t["text"] for t in tokens if t["text"]),
        "tokens": tokens,
        "layout_blocks": layout_blocks,
        "raw": payload,
    }


def specialized_ocr_layout(image_path: str, provider: str = "documentai") -> dict[str, Any]:
    provider_normalized = (provider or "documentai").strip().lower()
    if provider_normalized in {"documentai", "google_document_ai", "gcp"}:
        return ocr_layout_google_document_ai(image_path)
    if provider_normalized in {"textract", "aws_textract", "aws"}:
        return ocr_layout_aws_textract(image_path)
    if provider_normalized in {"azure", "azure_docint", "azure_document_intelligence"}:
        return ocr_layout_azure_docint(image_path)
    raise ValueError("Unsupported specialized OCR provider. Use: documentai, textract, azure")


def gemini_reason_over_ocr_layout(
    ocr_payload: dict[str, Any],
    document_type_hint: str = "UNKNOWN",
    model_name: str = "gemini-2.5-flash",
) -> dict[str, Any]:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError("Please install: pip install google-generativeai") from exc

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for reasoning layer")

    genai.configure(api_key=api_key)  # type: ignore[attr-defined]
    model = genai.GenerativeModel(model_name)  # type: ignore[attr-defined]

    tokens = ocr_payload.get("tokens", []) or []
    blocks = ocr_payload.get("layout_blocks", []) or []
    compact_tokens = tokens[:350]
    compact_blocks = blocks[:120]

    prompt = (
        "You are an enterprise IDP reasoning layer. "
        "Return ONLY strict JSON, no markdown. "
        "Use OCR tokens + layout blocks to produce normalized extraction. "
        "Schema: "
        "{\"document_type\": string, \"confidence_score\": float, "
        "\"kv\": object, \"table\": array, \"full_text\": string, \"review_reasons\": array}. "
        "If uncertain, keep fields empty and add review_reasons entries.\n\n"
        f"Document type hint: {document_type_hint}\n"
        f"OCR provider: {ocr_payload.get('provider', 'unknown')}\n"
        f"OCR full text: {ocr_payload.get('full_text', '')}\n"
        f"OCR tokens sample: {json.dumps(compact_tokens, ensure_ascii=True)}\n"
        f"Layout blocks sample: {json.dumps(compact_blocks, ensure_ascii=True)}"
    )

    response = model.generate_content(prompt)
    raw = _strip_json_fences(getattr(response, "text", "") or "")
    parsed = json.loads(raw)

    return {
        "document_type": str(parsed.get("document_type", document_type_hint) or document_type_hint),
        "confidence_score": _safe_float(parsed.get("confidence_score", 0.0), 0.0),
        "kv": parsed.get("kv", {}) if isinstance(parsed.get("kv", {}), dict) else {},
        "table": parsed.get("table", []) if isinstance(parsed.get("table", []), list) else [],
        "full_text": str(parsed.get("full_text", ocr_payload.get("full_text", "")) or ""),
        "review_reasons": parsed.get("review_reasons", []) if isinstance(parsed.get("review_reasons", []), list) else [],
        "raw": parsed,
    }


async def _html_to_png_async(html: str, output_path: str) -> bool:
    """Render an HTML string to PNG using Playwright Async API."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            page = await browser.new_page(viewport={"width": 900, "height": 1200})
            await page.set_content(html, wait_until="networkidle")
            # Resize to content height so cards are not clipped.
            content_height = await page.evaluate("() => document.body.scrollHeight")
            await page.set_viewport_size({"width": 900, "height": max(400, int(content_height) + 60)})
            await page.screenshot(path=output_path, full_page=True)
        finally:
            await browser.close()

    return True


def _html_to_png(html: str, output_path: str) -> bool:
    """Sync wrapper that safely executes async Playwright rendering."""
    try:
        try:
            asyncio.get_running_loop()
            in_event_loop = True
        except RuntimeError:
            in_event_loop = False

        if not in_event_loop:
            return asyncio.run(_html_to_png_async(html, output_path))

        # If we're already in FastAPI's loop, render in a worker thread.
        result: dict[str, Any] = {"ok": False, "err": None}

        def _runner() -> None:
            try:
                result["ok"] = asyncio.run(_html_to_png_async(html, output_path))
            except Exception as exc:  # pragma: no cover - defensive bridge code
                result["err"] = exc

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join()

        if result["err"]:
            raise result["err"]

        return bool(result["ok"])
    except Exception as exc:
        print(f"[RENDER] Playwright render failed: {exc}")
        return False


def _generate_with_imagen(
    prompt: str,
    output_path: str,
) -> Optional[str]:
    """
    Generate an image with Vertex AI Imagen 3 and save to output_path.
    Returns output_path on success, None when Imagen is unavailable
    (billing disabled, API not enabled, or SDK not installed).
    """
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1").strip()
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

    if not project:
        print("[IMAGEN] GOOGLE_CLOUD_PROJECT not set, skipping Imagen")
        return None

    # Resolve relative credential paths from the project root
    if creds_path and not os.path.isabs(creds_path):
        creds_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), creds_path
        )
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

    try:
        import vertexai
        from vertexai.preview.vision_models import ImageGenerationModel
    except ImportError:
        print("[IMAGEN] google-cloud-aiplatform not installed, skipping Imagen")
        return None

    try:
        vertexai.init(project=project, location=location)
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        print("[IMAGEN] Generating image with Imagen 3…")
        resp = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="3:4",
        )
        if not resp.images:
            print("[IMAGEN] No images returned")
            return None
        resp.images[0].save(output_path)
        print(f"[IMAGEN] Saved: {output_path} ({os.path.getsize(output_path)} bytes)")
        return output_path
    except Exception as exc:
        err_str = str(exc)
        if "BILLING_DISABLED" in err_str or "billing" in err_str.lower():
            print("[IMAGEN] Billing not enabled on GCP project — skipping Imagen")
        elif "403" in err_str or "PermissionDenied" in err_str:
            print(f"[IMAGEN] Permission denied — ensure service account has 'Vertex AI User' role: {exc}")
        else:
            print(f"[IMAGEN] Failed ({type(exc).__name__}): {exc}")
        return None


def generate_digital_form_image_gemini(
    extracted_text: str,
    doc_type: str,
    output_path: str,
) -> Optional[str]:
    """
    Pipeline:
      Extracted text (from OCR/Gemini)
        → [Priority 1] Vertex AI Imagen 3  (if billing enabled)
        → [Priority 2] Gemini generates HTML card → Playwright renders → PNG

    Returns output_path on success, None on failure (caller falls back to PIL).
    Env vars: GEMINI_API_KEY  (HTML path)
             GOOGLE_CLOUD_PROJECT + GOOGLE_APPLICATION_CREDENTIALS  (Imagen path)
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    doc_type_label = str(doc_type or "document").strip()

    # ── Priority 1: Vertex AI Imagen 3 ────────────────────────────────────
    imagen_out = output_path.replace(".png", "_imagen.png") if output_path.endswith(".png") else output_path + "_imagen.png"
    imagen_prompt = (
        f"A professional digital document card. Document type: {doc_type_label}. "
        "Clean white background, dark navy header bar, neat rows of text. "
        "Footer text: 'Digitized by DOC-INTEL AI'. "
        "Content summary: " + (extracted_text[:800] if extracted_text else "No content available.")
    )
    imagen_result = _generate_with_imagen(imagen_prompt, imagen_out)
    if imagen_result:
        return imagen_result

    # ── Priority 2: Gemini HTML → Playwright PNG ───────────────────────────
    if not api_key:
        print("[RENDER] GEMINI_API_KEY not set, skipping HTML render")
        return None

    prompt = f"""You are a professional UI generator and document designer.

Convert the following extracted text into a clean, modern, well-structured HTML card.

Document type: {doc_type_label}

Extracted content:
{extracted_text}

Rules:
- Return ONLY valid HTML — no markdown, no explanation, no code fences
- Use a single self-contained <div> root with all styles inline or in a <style> block
- White background, professional sans-serif font (system-ui, Arial)
- Width: 860px, auto height, padding 40px
- Add a branded header bar (dark navy #1e293b) with the document type as title
- List every line as a clean row — label in bold if key:value format, otherwise plain text
- Alternate row background (#f8fafc / #ffffff) for readability
- Add a subtle footer: "Digitized by DOC-INTEL AI"
- No external URLs, no images, no JavaScript
"""

    html_output: Optional[str] = None

    # Model preference list — first available/non-overloaded one wins
    _RENDER_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    def _call_genai_new(model_name: str) -> Optional[str]:
        """Try new google-genai SDK."""
        import time
        try:
            from google import genai as gai
            client = gai.Client(api_key=api_key)
            for attempt in range(3):
                try:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    return (resp.text or "").strip()
                except Exception as e:
                    err = str(e)
                    if "503" in err or "UNAVAILABLE" in err or "overloaded" in err.lower():
                        wait = 2 ** attempt
                        print(f"[RENDER] {model_name} busy (attempt {attempt+1}), retrying in {wait}s…")
                        time.sleep(wait)
                        continue
                    raise
        except ImportError:
            pass
        except Exception as exc:
            print(f"[RENDER] google-genai {model_name} failed ({type(exc).__name__}): {exc}")
        return None

    def _call_genai_legacy(model_name: str) -> Optional[str]:
        """Try legacy google-generativeai SDK as fallback."""
        import time
        try:
            import google.generativeai as genai_legacy  # type: ignore
            genai_legacy.configure(api_key=api_key)
            model = genai_legacy.GenerativeModel(model_name)
            for attempt in range(3):
                try:
                    resp = model.generate_content(prompt)
                    return (resp.text or "").strip()
                except Exception as e:
                    err = str(e)
                    if "503" in err or "UNAVAILABLE" in err or "overloaded" in err.lower():
                        wait = 2 ** attempt
                        print(f"[RENDER] legacy {model_name} busy (attempt {attempt+1}), retrying in {wait}s…")
                        time.sleep(wait)
                        continue
                    raise
        except ImportError:
            pass
        except Exception as exc:
            print(f"[RENDER] legacy genai {model_name} failed ({type(exc).__name__}): {exc}")
        return None

    def _clean_html(raw: str) -> Optional[str]:
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        return raw if raw.startswith("<") else None

    for model_name in _RENDER_MODELS:
        print(f"[RENDER] Trying HTML generation with {model_name}…")
        raw = _call_genai_new(model_name)
        if not raw:
            raw = _call_genai_legacy(model_name)
        if raw:
            html_output = _clean_html(raw)
            if html_output:
                print(f"[RENDER] HTML generated by {model_name} ({len(html_output)} chars)")
                break
            else:
                print(f"[RENDER] {model_name} did not return HTML, got: {(raw or '')[:80]}")

    if not html_output:
        print("[RENDER] Gemini returned no usable HTML")
        return None

    print(f"[RENDER] Gemini HTML generated ({len(html_output)} chars), rendering with Playwright…")
    success = _html_to_png(html_output, output_path)
    if success and os.path.exists(output_path):
        print(f"[RENDER] PNG saved: {output_path}")
        return output_path

    return None


def run_enterprise_idp_pipeline(
    image_path: str,
    ocr_provider: str = "documentai",
    reasoning_provider: str = "gemini",
    document_type_hint: str = "UNKNOWN",
) -> dict[str, Any]:
    if (reasoning_provider or "gemini").strip().lower() != "gemini":
        raise ValueError("Only reasoning_provider='gemini' is currently supported")

    ocr_payload = specialized_ocr_layout(image_path=image_path, provider=ocr_provider)
    reasoning = gemini_reason_over_ocr_layout(
        ocr_payload=ocr_payload,
        document_type_hint=document_type_hint,
    )

    model_runs = [
        {
            "stage": "ocr_layout",
            "model_name": ocr_provider,
            "provider": ocr_provider,
            "success": "true",
            "duration_ms": 0.0,
            "raw_output": {
                "token_count": len(ocr_payload.get("tokens", []) or []),
                "layout_block_count": len(ocr_payload.get("layout_blocks", []) or []),
            },
        },
        {
            "stage": "reasoning",
            "model_name": "gemini-2.5-flash",
            "provider": "gemini",
            "success": "true",
            "duration_ms": 0.0,
            "raw_output": reasoning.get("raw", {}),
        },
    ]

    return {
        "doc_analysis": {
            "type": reasoning.get("document_type", "UNKNOWN"),
            "confidence_score": reasoning.get("confidence_score", 0.0),
        },
        "data": {
            "full_text": reasoning.get("full_text", ""),
            "kv": reasoning.get("kv", {}),
            "table": reasoning.get("table", []),
            "review_reasons": reasoning.get("review_reasons", []),
        },
        "ocr_results": _normalize_tokens_to_legacy_words(ocr_payload.get("tokens", []) or []),
        "layout_blocks": ocr_payload.get("layout_blocks", []) or [],
        "source_engine": f"{ocr_provider}+gemini",
        "model_runs": model_runs,
        "ocr_payload": ocr_payload,
    }
