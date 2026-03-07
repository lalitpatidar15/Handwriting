# 🚀 VANGUARD INTELLIGENCE | AI Command Center

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OCR](https://img.shields.io/badge/OCR-DocTR%20%7C%20TrOCR-orange)](https://github.com/mindee/doctr)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)](https://streamlit.io/)

Vanguard Intelligence is an enterprise-grade OCR and Handwriting Recognition system designed for high-accuracy digitization of complex handwritten notes, medical prescriptions, and structured forms. It features a hybrid local-first architecture with optional Cloud Super Mode (Gemini/GPT-4).

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

To use **Cloud Super Mode**, you can either enter your keys in the Sidebar UI or set them as environment variables:
-   `GEMINI_API_KEY`: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
-   `OPENAI_API_KEY`: Get from [OpenAI Dashboard](https://platform.openai.com/api-keys)

---
*Developed for High-Stakes Document Intelligence.*
