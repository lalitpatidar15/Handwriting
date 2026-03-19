# Enterprise IDP Architecture

Version: 1.1
Target: Specialized-OCR + Gemini-reasoning enterprise Intelligent Document Processing platform
Scope: handwritten documents, invoices, prescriptions, hand-filled forms, template learning, exact reconstruction, human review

## 1. Objective

The current codebase already covers a useful baseline:

- Streamlit frontend in `app.py`
- FastAPI backend in `api.py`
- local OCR and vision in `ocr_engine.py` and `vision_engine.py`
- cloud fallback in `cloud_ocr.py`
- document parsing in `universal_parser.py`

The next-generation target is not a single OCR pipeline. It is a layered IDP platform that combines:

- layout-aware OCR
- Gemini reasoning
- domain-specific validation
- template memory
- human review
- continuous learning

The realistic path to `95%` to `99%` enterprise accuracy is not one model. It is a controlled ensemble with confidence gating.

## 2. Target Principles

1. Gemini should be the reasoning layer, not the only extraction layer.
2. Bounding boxes and layout metadata must be preserved from the first stage.
3. Every extracted field needs provenance, confidence, and validation status.
4. Low-confidence output must route to human review instead of forcing automation.
5. Corrected outputs must feed template memory and prompt adaptation.
6. Domain engines should be isolated so invoice rules do not pollute prescription logic.

## 3. Production Architecture

```text
Client Apps
(Web / Mobile / Partner API)
        |
        v
API Gateway
(Auth, quotas, tenant routing)
        |
        v
Document Ingestion Service
        |
        +--> Object Storage
        |
        +--> Metadata Store
        |
        v
Workflow Orchestrator / Queue
        |
        v
Document Intelligence Pipeline
        |
        +--> Preprocessing Service
        +--> Layout Detection Service
        +--> OCR Ensemble Service
        +--> Document Classification Service
        +--> Domain Intelligence Engine
        +--> Gemini Reasoning Layer
        +--> Confidence & Validation Engine
        +--> Reconstruction Engine
        +--> Review Task Generator
        |
        v
Structured Data Store
        |
        +--> Analytics API
        +--> Review UI
        +--> Learning / Template Engine
        +--> Vector Memory
```

## 4. Core Services

### 4.1 API Gateway

Responsibilities:

- tenant authentication
- API key validation
- rate limiting
- audit logging
- request tracing

Recommended stack:

- Kong, NGINX, or managed API gateway
- JWT or OAuth2 per tenant

### 4.2 Document Ingestion Service

Responsibilities:

- accept uploads from web, mobile, or API
- store original document in object storage
- create a `document_id`
- compute hash for deduplication
- enqueue processing job

Minimal metadata:

```json
{
  "document_id": "doc_8f2d",
  "tenant_id": "tenant_12",
  "file_name": "invoice_1203.jpg",
  "mime_type": "image/jpeg",
  "sha256": "...",
  "source": "web_upload",
  "status": "queued"
}
```

### 4.3 Object Storage

Store:

- original image or PDF
- cleaned image
- cropped regions
- reconstructed HTML or PDF
- human-reviewed final output

Recommended paths:

```text
/tenant/{tenant_id}/documents/{document_id}/original
/tenant/{tenant_id}/documents/{document_id}/processed
/tenant/{tenant_id}/documents/{document_id}/regions
/tenant/{tenant_id}/documents/{document_id}/reconstructed
/tenant/{tenant_id}/documents/{document_id}/reviewed
```

### 4.4 Workflow Orchestrator

Responsibilities:

- asynchronous processing
- retries
- timeout management
- multi-stage fan-out and fan-in
- human review routing

Recommended stack:

- Celery + Redis for a fast start
- Temporal for enterprise-grade workflow durability
- RabbitMQ or Kafka if volume grows

## 5. Document Intelligence Pipeline

### 5.1 Preprocessing Service

Goal:

- normalize noisy images before OCR
- retain a copy of the original for reconstruction

Extend the current `vision_engine.py` to include:

- skew correction
- perspective correction
- shadow removal
- stamp and line enhancement
- page segmentation for multi-page PDFs
- ink-preserving binarization for handwriting

Output:

- cleaned page image
- transformation matrix
- preprocessing metrics

### 5.2 Layout Detection Service

Goal:

- identify document zones before text extraction

Detect:

- headers
- tables
- line items
- handwritten regions
- checkboxes
- stamps
- signatures
- key-value regions

Recommended models:

- LayoutLMv3 or Detectron-style layout detector for structured layouts
- DocTR region outputs for lightweight baseline
- Gemini vision for hard cases or ambiguous layouts

Output schema:

```json
{
  "page": 1,
  "blocks": [
    {
      "block_id": "blk_1",
      "type": "table",
      "bbox": [90, 140, 760, 1080],
      "confidence": 0.96
    },
    {
      "block_id": "blk_2",
      "type": "handwritten_note",
      "bbox": [88, 1090, 770, 1320],
      "confidence": 0.91
    }
  ]
}
```

### 5.3 OCR Ensemble Service

Goal:

- extract text with provenance and box-level confidence

Recommended strategy:

- printed text: DocTR or Tesseract fallback
- handwriting: TrOCR or Florence-style handwritten model
- hard regions: Gemini vision crop-level extraction

Do not rely on one OCR result. Use a voting and reconciliation layer.

Per-token record:

```json
{
  "text": "Paracetamol",
  "bbox": [122, 418, 280, 451],
  "source_engine": "trocr",
  "ocr_confidence": 0.87,
  "page": 1,
  "region_id": "blk_2"
}
```

### 5.4 Document Classification Service

Goal:

- classify the document before domain extraction

Target classes:

- handwritten_note
- printed_invoice
- handwritten_invoice
- prescription
- hand_filled_form
- mixed_document

Inputs:

- layout blocks
- OCR summary
- document embeddings
- template similarity matches

Gemini usage:

- final arbitration for ambiguous cases only

### 5.5 Domain Intelligence Engines

Each domain gets its own schema, prompts, validators, and review rules.

#### Invoice Intelligence

Fields:

- invoice_number
- supplier_name
- supplier_tax_id
- invoice_date
- due_date
- currency
- subtotal
- tax
- total
- line_items

Rules:

- `subtotal + tax ~= total`
- line item totals should reconcile with invoice total
- supplier name should match known vendor memory if available
- duplicate invoice number per supplier should trigger fraud review

#### Prescription Intelligence

Fields:

- patient_name
- diagnosis
- medicines
- dosage
- route
- frequency
- duration
- doctor_name
- notes

Rules:

- medicine names cross-checked with a medication reference source
- dosage and frequency normalized to canonical format
- cause and effect stored as medical knowledge annotations, not as verified clinical truth

Suggested annotation model:

```json
{
  "name": "Paracetamol",
  "dose": "500 mg",
  "frequency": "TDS",
  "duration": "3 days",
  "normalized_frequency": "3 times daily",
  "therapeutic_use": "reduces fever and pain",
  "confidence": 0.92
}
```

#### Handwritten Document Engine

Outputs:

- paragraph reconstruction
- bullet detection
- action item extraction
- named entities
- optional summary

#### Form Intelligence

Outputs:

- labels
- values
- checkboxes
- signature presence
- field coordinates

Rules:

- preserve original field positions
- distinguish printed template text from handwritten input

## 6. Gemini Reasoning Layer

Gemini should operate as a constrained structured reasoning layer.

Use Gemini for:

- schema-constrained extraction
- ambiguous handwriting interpretation using crop images and OCR candidates
- table understanding when local heuristics fail
- medical normalization and dosage interpretation
- template classification and field alignment
- natural language explanation for review tasks

Do not use Gemini as an unrestricted free-text parser. Wrap it with:

- JSON schema output contracts
- candidate OCR text lists
- bounding boxes
- template priors
- validation instructions

Recommended prompt inputs:

1. page image or region crop
2. OCR candidates from local engines
3. detected layout blocks
4. expected schema for the domain
5. template hints from vector retrieval
6. strict extraction rules and refusal policy

Recommended prompt outputs:

```json
{
  "document_type": "prescription",
  "fields": [
    {
      "name": "diagnosis",
      "value": "Fever",
      "confidence": 0.94,
      "evidence": ["blk_3", "ocr_tok_18"]
    }
  ],
  "line_items": [],
  "review_reasons": []
}
```

## 7. Confidence and Validation Engine

This layer is mandatory if the goal is enterprise accuracy.

Confidence should be built from multiple signals:

- OCR confidence
- Gemini confidence
- template match score
- validator pass or fail
- field consistency with neighboring fields
- vendor or doctor prior history
- human correction history on similar documents

Example field confidence model:

```text
final_field_confidence =
0.30 * ocr_confidence +
0.25 * llm_confidence +
0.20 * template_alignment_score +
0.15 * rule_validation_score +
0.10 * historical_prior_score
```

Suggested thresholds:

- above `0.92`: auto-approve
- `0.75` to `0.92`: send field-level review
- below `0.75`: reprocess with alternate extraction strategy and then queue review

Validation examples:

- invoice date must parse to valid date
- quantity must be numeric
- total must reconcile
- medicine name must exist in reference index or fuzzy match to known synonyms

## 8. Human Review System

The review interface should not just show raw JSON. It should show:

- original document image
- overlay bounding boxes
- extracted fields with confidence colors
- suggested corrections
- validation errors
- approval audit trail

Review modes:

- document-level review
- field-level review
- fraud review
- template approval review

Store reviewer actions:

```json
{
  "review_id": "rev_102",
  "document_id": "doc_8f2d",
  "field_name": "invoice_number",
  "predicted_value": "INV-1208",
  "corrected_value": "INV-1203",
  "reviewer_id": "user_17",
  "reason": "OCR confusion between 8 and 3"
}
```

## 9. Learning and Template Engine

### 9.1 Automatic Template Learning

Each document should produce a structural fingerprint built from:

- logo region position
- header block layout
- label positions
- table geometry
- relative distances between fields

Template fingerprint example:

```json
{
  "template_id": "tpl_retail_001",
  "tenant_id": "tenant_12",
  "document_type": "printed_invoice",
  "fingerprint": {
    "page_size": [827, 1169],
    "anchors": [
      {"label": "Invoice Number", "bbox": [102, 118, 228, 154]},
      {"label": "Date", "bbox": [492, 121, 562, 151]},
      {"label": "Total", "bbox": [480, 1015, 564, 1042]}
    ],
    "table_region": [90, 260, 740, 930]
  },
  "sample_count": 26,
  "approval_status": "approved"
}
```

When a new document arrives:

1. compute structural embedding
2. search nearest templates
3. if similarity is high, use template-guided extraction
4. if similarity is low, route to cold-start path and consider learning a new template

### 9.2 Continuous Learning Loop

Use reviewed corrections for:

- prompt examples
- template field refinements
- field synonym dictionaries
- OCR post-correction dictionaries
- confidence recalibration

In the early stages, do not fine-tune Gemini. Improve:

- retrieval
- prompts
- validators
- template priors
- OCR routing

## 10. Exact Layout Reconstruction

To achieve digital twins, preserve:

- text
- bounding boxes
- font hints if inferred
- line segments
- checkbox states
- signature regions
- original page size

Core reconstruction record:

```json
{
  "page": 1,
  "width": 827,
  "height": 1169,
  "elements": [
    {
      "type": "text",
      "text": "Invoice Number",
      "bbox": [102, 118, 228, 154],
      "source": "template"
    },
    {
      "type": "text",
      "text": "INV-1203",
      "bbox": [240, 118, 360, 154],
      "source": "handwritten_fill"
    }
  ]
}
```

Output formats:

- absolutely positioned HTML
- searchable PDF
- JSON with coordinates
- redaction-ready overlay format

## 11. Fraud Detection Layer

Best applied initially to invoices.

Signals:

- duplicate invoice number
- duplicate total for same supplier over small interval
- suspicious bank or GST changes
- outlier totals against supplier history
- template mismatch for claimed supplier
- edited line item regions or tampering cues

Fraud output:

```json
{
  "fraud_risk": "medium",
  "score": 0.67,
  "rules_triggered": [
    "duplicate_invoice_number",
    "supplier_template_mismatch"
  ]
}
```

## 12. Vector Document Memory

Use vector memory for:

- template retrieval
- similar document search
- duplicate detection
- vendor and doctor clustering
- prompt example retrieval

Recommended collections:

- document embeddings
- template embeddings
- supplier profile embeddings
- correction example embeddings

Good options:

- PostgreSQL + pgvector for simplest architecture
- Weaviate for richer semantic search
- Elasticsearch if you also want keyword plus vector hybrid search

## 13. Multi-Tenant Data Model

Required top-level entities:

- tenants
- documents
- pages
- extraction_blocks
- templates
- review_tasks
- field_predictions
- field_corrections
- model_runs
- fraud_alerts

Critical isolation rules:

- per-tenant encryption keys where possible
- tenant-specific prompt examples
- tenant-specific template memory
- tenant-specific review queues

## 14. Suggested API Contracts

### 14.1 Submit document

```text
POST /v1/documents
```

Returns:

```json
{
  "document_id": "doc_8f2d",
  "status": "queued"
}
```

### 14.2 Get result

```text
GET /v1/documents/{document_id}
```

### 14.3 Get review tasks

```text
GET /v1/review/tasks
```

### 14.4 Submit correction

```text
POST /v1/review/tasks/{task_id}/complete
```

### 14.5 Retrieve template match

```text
GET /v1/templates/{template_id}
```

## 15. Accuracy Strategy for 95% to 99%

This target is feasible only with the right measurement scope.

Expected path:

- generic cold-start extraction across mixed docs: `75%` to `88%`
- after template learning and validators: `88%` to `94%`
- after tenant-specific review loop and retrieval: `94%` to `98%`
- stable narrow document families with strong review feedback: `98%` plus on key fields

What drives the final jump:

- template reuse
- field-level validators
- domain dictionaries
- human review feedback
- per-tenant priors
- structured Gemini prompts with evidence

## 16. Mapping to the Current Repository

Current file to future role mapping:

- `app.py` -> temporary operator UI, later split into review UI and admin dashboard
- `api.py` -> ingestion API and orchestration facade
- `vision_engine.py` -> preprocessing service
- `ocr_engine.py` -> OCR ensemble coordinator
- `cloud_ocr.py` -> Gemini and external LLM connector
- `universal_parser.py` -> seed for document classification and initial structured parsing
- `medical_ai.py` -> prescription knowledge normalization
- `models.py` and `database.py` -> early metadata store, later expand to production schema

## 17. Recommended Implementation Phases

### Phase 1: Stabilize current system

- persist page-level OCR tokens with bounding boxes
- add structured result schema versioning
- store raw Gemini outputs separately from final normalized outputs
- add review status to each document

### Phase 2: Introduce review and confidence

- add `field_predictions` and `review_tasks` tables
- compute confidence per field
- create basic review UI for corrections

### Phase 3: Add template memory

- store anchor labels and coordinates
- create template similarity matcher
- use template hints in extraction prompts

### Phase 4: Add domain engines

- invoice engine
- prescription engine
- hand-filled form engine
- handwritten note engine

### Phase 5: Add enterprise controls

- multi-tenancy
- vector memory
- fraud detection
- analytics dashboards

## 18. Minimum New Tables

Suggested additions beyond the current `documents` table:

```text
templates
document_pages
extraction_blocks
field_predictions
review_tasks
field_corrections
model_runs
fraud_alerts
tenant_settings
```

## 19. Build Recommendation

For this repo, the most practical next build step is:

1. keep FastAPI as the control plane
2. keep local OCR for first-pass extraction
3. use Gemini for constrained reasoning and fallback extraction
4. add field-level confidence and review workflows before chasing more model complexity
5. add template memory before attempting model fine-tuning

That sequence will move the system closest to ABBYY, Hyperscience, and UiPath-style operational accuracy with the least wasted effort.

## 20. Implemented Backend Delta in This Repository

The backend now includes an enterprise processing mode in the existing FastAPI API:

- `POST /api/v1/process_document` supports:
  - `enterprise_mode=true`
  - `enterprise_ocr_provider=documentai|textract|azure`
  - `enterprise_reasoning_provider=gemini`
- `GET /api/v1/providers/requirements` returns machine-readable credential requirements.

Current implementation behavior:

1. preprocess image using local vision enhancement
2. call specialized OCR + layout provider
3. normalize OCR tokens and layout blocks into a shared schema
4. call Gemini for constrained JSON reasoning only
5. persist confidence, review tasks, template match, model runs

Additional persisted metadata on each document:

- `layout_provider`
- `reasoning_provider`

## 21. Required API Credentials

Use the following credentials depending on selected OCR provider.

### 21.1 Reasoning Layer (mandatory in enterprise mode)

- `GEMINI_API_KEY`

### 21.2 OCR + Layout Provider Options

Google Document AI:

- `GOOGLE_DOCUMENT_AI_PROJECT_ID`
- `GOOGLE_DOCUMENT_AI_LOCATION` (example: `us`)
- `GOOGLE_DOCUMENT_AI_PROCESSOR_ID`
- `GOOGLE_DOCUMENT_AI_API_KEY`

AWS Textract:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (example: `us-east-1`)

Azure AI Document Intelligence:

- `AZURE_DOCINT_ENDPOINT`
- `AZURE_DOCINT_API_KEY`
- `AZURE_DOCINT_API_VERSION` (optional, default: `2024-07-31-preview`)

## 22. Minimal Environment Example

```bash
# reasoning
export GEMINI_API_KEY="your_gemini_key"

# choose one OCR provider block below

# Google Document AI
export GOOGLE_DOCUMENT_AI_PROJECT_ID="your_project_id"
export GOOGLE_DOCUMENT_AI_LOCATION="us"
export GOOGLE_DOCUMENT_AI_PROCESSOR_ID="your_processor_id"
export GOOGLE_DOCUMENT_AI_API_KEY="your_document_ai_key"

# AWS Textract
export AWS_ACCESS_KEY_ID="your_aws_access_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
export AWS_REGION="us-east-1"

# Azure Document Intelligence
export AZURE_DOCINT_ENDPOINT="https://<resource>.cognitiveservices.azure.com"
export AZURE_DOCINT_API_KEY="your_azure_docint_key"
export AZURE_DOCINT_API_VERSION="2024-07-31-preview"
```

## 23. MongoDB Atlas-First Database Strategy

For AI-native IDP workloads, MongoDB Atlas can simplify the stack by combining:

- document metadata storage
- AI extraction JSON storage
- processing logs
- template collections
- vector embeddings via Atlas Vector Search

This allows a practical consolidation where MongoDB replaces separate relational + vector + search systems for many workloads.

### 23.1 Recommended Collections

- `documents`
- `extraction_results`
- `ai_processing_logs`
- `templates`
- `review_corrections`
- `embeddings`

### 23.2 Implemented Mongo Endpoints in This Repository

- `GET /api/v1/mongo/health`
- `GET /api/v1/mongo/documents/{document_id}`
- `POST /api/v1/mongo/vector/search`

### 23.3 Environment Variables

- `MONGODB_URI`
- `MONGODB_DB_NAME`
- `MONGODB_VECTOR_INDEX`

### 23.4 Notes

- Existing SQL tables remain available for compatibility and phased rollout.
- Mongo mirroring runs after successful document processing and review updates.
- The current embedding pipeline uses deterministic text vectors as a bootstrap path; replace with model-based embeddings for production-grade semantic retrieval.

### 23.5 Connection Prerequisites

Before the server can reach MongoDB Atlas, two conditions must be met:

1. **IP whitelist**: Your machine's IP must be added to MongoDB Atlas → Network Access → Add IP Address. If the IP is not whitelisted, the connection will fail with an SSL/TLS internal error at the handshake stage — this is how Atlas blocks unauthorized IPs before any authentication is attempted. Use `0.0.0.0/0` for development or your specific IP for production.

2. **pymongo installed**: Run `pip install -r requirements.txt` inside the project virtual environment. The system Python will not have `pymongo`.

If `GET /api/v1/mongo/health` returns `"connected": false`, check these two items first.