import base64
import os
import json
import re

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
    
    genai.configure(api_key=api_key)
    
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
            model = genai.GenerativeModel(model_name)
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
            
    raise last_error

def cloud_ocr_openai(image_path):
    """
    OpenAI GPT-4o Vision Path (Paid).
    Provides human-level transcription for complex handwriting.
    """
    from openai import OpenAI
    
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
