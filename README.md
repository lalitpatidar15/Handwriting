# 🚀 VANGUARD INTELLIGENCE | AI Command Center

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OCR](https://img.shields.io/badge/OCR-DocTR%20%7C%20TrOCR-orange)](https://github.com/mindee/doctr)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)](https://streamlit.io/)

Vanguard Intelligence is an enterprise-grade OCR and Handwriting Recognition system designed for high-accuracy digitization of complex handwritten notes, medical prescriptions, and structured forms. It features a hybrid local-first architecture with optional Cloud Super Mode (Gemini/GPT-4).

For the enterprise target-state architecture and implementation roadmap, see `ENTERPRISE_IDP_ARCHITECTURE.md`.

## 🛡️ Core Features

-   **Hybrid OCR Engine**: Combines **DocTR** (Layout Perception) and **TrOCR** (Deep Handwriting Analysis).
-   **Cloud Super Mode**: Native integration with **Gemini 2.5 Flash** (FREE) and **GPT-4o Vision** (Paid).
-   **Intelligent Parsing**: Automatic detection of tables (`STRUCTURED_FORM`) vs. notes (`HANDWRITTEN_NOTE`).
-   **Medical Layer**: Specialized fuzzy correction for healthcare documentation.
-   **Vision Pipeline**: Advanced OpenCV-based ink-sharpening and contrast normalization.

## 🛠️ Technology Stack

-   **Frontend**: Streamlit (Premium Custom UI)
-   **Primary OCR**: `python-doctr` (MobileNet + CRNN)
-   **Secondary/Refinement OCR**: `transformers` (microsoft/trocr-large)
-   **Cloud Intelligence**: `google-generativeai`, `openai`
-   **Image Processing**: `opencv-python`, `Pillow`
-   **Fuzzy Logic**: `rapidfuzz`

## 🚀 Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone <your-repo-url>
    cd <repo-folder>
    ```

2.  **Setup Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the System**:
    ```bash
    streamlit run app.py
    ```

## ☁️ Cloud Configuration

Cloud and enterprise credentials are configured on the backend via environment variables only.
The frontend does not collect or store provider keys.

### Enterprise OCR + Reasoning Configuration

This repository now supports an enterprise backend mode where OCR/layout is handled by specialized OCR APIs and Gemini is used for constrained reasoning.

Set these environment variables depending on the OCR provider:

- Google Document AI:
    - `GOOGLE_DOCUMENT_AI_PROJECT_ID`
    - `GOOGLE_DOCUMENT_AI_LOCATION` (example: `us`)
    - `GOOGLE_DOCUMENT_AI_PROCESSOR_ID`
    - `GOOGLE_DOCUMENT_AI_API_KEY`
- AWS Textract:
    - `AWS_ACCESS_KEY_ID`
    - `AWS_SECRET_ACCESS_KEY`
    - `AWS_REGION` (example: `us-east-1`)
- Azure AI Document Intelligence:
    - `AZURE_DOCINT_ENDPOINT`
    - `AZURE_DOCINT_API_KEY`
    - `AZURE_DOCINT_API_VERSION` (optional, default: `2024-07-31-preview`)
- Reasoning layer:
    - `GEMINI_API_KEY`

### MongoDB Atlas Configuration (Main DB + Vector Search)

The backend now supports MongoDB Atlas for document metadata, extraction outputs, processing logs, templates, review corrections, and vector embeddings.

- `MONGODB_URI`
- `MONGODB_DB_NAME`
- `MONGODB_VECTOR_INDEX`

### Enterprise API usage

Call `POST /api/v1/process_document` with:

- `enterprise_mode=true`
- `enterprise_ocr_provider=documentai|textract|azure`
- `enterprise_reasoning_provider=gemini`

To fetch a machine-readable list of required credentials:

- `GET /api/v1/providers/requirements`

To validate what is configured at runtime:

- `GET /api/v1/providers/health`

Mongo-specific endpoints:

- `GET /api/v1/mongo/health`
- `GET /api/v1/mongo/documents/{document_id}`
- `POST /api/v1/mongo/vector/search`

## Async Queue Mode (Celery)

This project now includes queue scaffolding for asynchronous enterprise processing.

Key files:

- `celery_app.py`
- `workers/document_tasks.py`
- `.env.example`

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Redis (example with Docker):

```bash
docker run -p 6379:6379 redis:7
```

Run worker:

```bash
celery -A celery_app.celery_app worker -l info
```

Async submit endpoint:

- `POST /api/v1/process_document/async`
    - accepts file upload plus:
        - `enterprise_ocr_provider=documentai|textract|azure`
        - `enterprise_reasoning_provider=gemini`

## Environment Template

Use `.env.example` as the starting point for local and deployment configuration.

---
*Developed for High-Stakes Document Intelligence.*
