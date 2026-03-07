# 📘 VANGUARD INTELLIGENCE | Developer & Handover Guide

Welcome to the technical core of the Vanguard Intelligence system. This document is designed for developers who will be maintaining, scaling, or refactoring the codebase.

## 🏗️ System Architecture

### 1. Vision Engine (`vision_engine.py`)
-   **Goal**: Prepare the image for OCR by maximizing ink contrast.
-   **Key Steps**: Grayscale → CLAHE (Local Contrast) → Median Blur → Sharpening (Unsharp Mask).
-   **Performance**: Resizes images to a standard height (800px) to balance OCR speed and accuracy.

### 2. OCR Engine (`ocr_engine.py`)
-   **DocTR (Primary)**: Uses `db_mobilenet_v3_large` (Detection) and `crnn_vgg16_bn` (Recognition). We switched to these lightweight models for sub-1s initialization.
-   **TrOCR (High-Fidelity)**: Selected only for "ink scan" mode. It is loaded *lazily* via `_load_troc_model` to prevent overhead on startup.
-   **Refinement Logic**: Only words with low DocTR confidence are sent to TrOCR to prevent widespread'hallucinations'.

### 3. Universal Parser (`universal_parser.py`)
-   **Geometric Logic**: Groups words into lines based on Y-coordinate centroids.
-   **Word Spacing**: Uses an adaptive threshold (30% of average word width) to detect gaps.
-   **Word Splitting**: `_fix_merged_words` uses regex to split common OCR merges (e.g., "Dontlet" → "Dont let").
-   **Classification**: Uses a density heuristic to decide if a document is a `STRUCTURED_FORM` (Table) or `HANDWRITTEN_NOTE` (Paragraph).

### 4. Cloud Super Mode (`cloud_ocr.py`)
-   **Native Integration**: Supports both Gemini (v2.0/2.5) and OpenAI (GPT-4o).
-   **Fallback Sequence**: Highly robust fallback logic ensures if one model fails, the next one is tried immediately.

## 🛠️ Performance Tuning

If accuracy is low:
1.  Check `vision_engine.py` sharpening weights.
2.  Enable `High-Fidelity Ink Scan` in the UI to trigger the TrOCR refinement layer.
3.  Use **Gemini 2.5 Flash** for the highest handwriting perception without cost.

## 🔮 Future Roadmap

-   **Fine-Tuning**: Train a custom TrOCR-small model on the specific handwriting styles of your users.
-   **PDF Support**: Add `pdf2image` integration to support multi-page digital documents.
-   **Database Integration**: Mirror the `kv` (Key-Value) results into a PostgreSQL or MongoDB instance for historical search.

---
*Maintained for Excellence.*
