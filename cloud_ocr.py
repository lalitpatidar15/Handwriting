import base64
import os
import json

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
    prompt = "Transcribe this handwritten document exactly as it is. Preserve the structure and lines. If it's a table, return the text line by line. Do not add any conversational text or explanations."

    for model_name in model_names:
        try:
            print(f"[OCR] Trying Gemini model: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, img])
            return response.text
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

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this handwritten document exactly as it is. Preserve the structure and lines. If it's a table, return the text line by line. Do not add any conversational text."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=1000,
    )

    return response.choices[0].message.content

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
