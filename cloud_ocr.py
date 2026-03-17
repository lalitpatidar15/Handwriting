import base64
import os
import json
import re
import time
from typing import Any

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
    FREE TIER: Google Gemini API for OCR.
    Free tier: 15 requests/minute, 1500 requests/day
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Please install: pip install google-generativeai")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Get free key from: https://aistudio.google.com/app/apikey")
    
    genai.configure(api_key=api_key)  # type: ignore[attr-defined]
    
    # Try multiple model names for robustness (Added 2.0/2.5 for peak performance)
    # The new API key supports 2.0 and 2.5 flash versions
    model_names = [
        'gemini-2.5-flash',
        'gemini-2.0-flash', 
        'gemini-1.5-flash', 
        'gemini-1.5-flash-latest'
    ]
    last_error = None
    
    import PIL.Image
    img = PIL.Image.open(image_path)
    prompt = """
    You are an Enterprise Intelligent Document Processing (IDP) AI. 
    Analyze this document (handwritten note, form, bill, or prescription) and extract the data strictly as a JSON object.
    
    Rules for output:
    1. Do NOT wrap the JSON in markdown code blocks like ```json ... ```. Just return raw JSON.
    2. Do NOT add any conversational text, explanations, or greetings.
    3. Use this exact schema format:
    {
      "document_type": "string (e.g., medical_prescription, invoice, handwritten_note, form)",
      "confidence_score": 0.95,
      "metadata": {
        "date": "extracted or null",
        "patient_or_client_name": "extracted or null"
      },
      "extracted_data": [
        // For tables/bills/prescriptions, put rows here as objects (e.g. {"medicine": "Amox", "dose": "500mg"})
        // For paragraphs, put paragraphs here as {"text": "..."}
      ],
      "full_raw_text": "string containing all the text"
    }
    """

    for model_name in model_names:
        try:
            print(f"[OCR] Trying Gemini model: {model_name}")
            model = genai.GenerativeModel(model_name)  # type: ignore[attr-defined]
            response = model.generate_content([prompt, img])
            
            # Clean up potential markdown formatting from Gemini
            text_response = response.text.strip()
            if text_response.startswith('```json'):
                text_response = text_response[7:]
            if text_response.startswith('```'):
                text_response = text_response[3:]
            if text_response.endswith('```'):
                text_response = text_response[:-3]
                
            text_response = text_response.strip()
            
            # Verify it's valid JSON
            try:
                json_data = json.loads(text_response)
                return json_data # Return the actual dict/json
            except json.JSONDecodeError:
                print(f"[OCR] Gemini {model_name} generated invalid JSON, returning raw text.")
                return {"full_raw_text": text_response, "document_type": "UNKNOWN"}
                
        except Exception as e:
            last_error = e
            print(f"[OCR] Gemini {model_name} failed: {e}")
            continue
            
    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini OCR failed for all model variants")


def cloud_ocr_gemini_structured(image_path):
    """Extract structured form content using Gemini and return canonical JSON dict."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Please install: pip install google-generativeai")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Get free key from: https://aistudio.google.com/app/apikey")

    genai.configure(api_key=api_key)  # type: ignore[attr-defined]

    model_names = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
    ]
    last_error = None

    import PIL.Image
    img = PIL.Image.open(image_path)
    prompt = (
        "You are a form extraction engine. Read this form image and return ONLY valid JSON with this schema: "
        "{\"document_type\": string, \"full_text\": string, \"fields\": [{\"label\": string, \"value\": string}], \"table\": [object]}. "
        "Rules: 1) Preserve field labels exactly. 2) Put handwritten or filled values in value. "
        "3) If value is empty, use empty string. 4) Do not include markdown or explanations."
    )

    for model_name in model_names:
        try:
            print(f"[OCR] Trying Gemini structured model: {model_name}")
            model = genai.GenerativeModel(model_name)  # type: ignore[attr-defined]
            response = model.generate_content([prompt, img])
            raw = _strip_json_fences(response.text)
            parsed = json.loads(raw)

            fields = parsed.get("fields", []) or []
            normalized_fields = []
            for item in fields:
                if isinstance(item, dict):
                    normalized_fields.append({
                        "label": str(item.get("label", "") or "").strip(),
                        "value": str(item.get("value", "") or "").strip(),
                    })

            table = parsed.get("table", [])
            if not isinstance(table, list):
                table = []

            return {
                "document_type": str(parsed.get("document_type", "STRUCTURED_FORM") or "STRUCTURED_FORM"),
                "full_text": str(parsed.get("full_text", "") or ""),
                "fields": normalized_fields,
                "table": table,
            }
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
