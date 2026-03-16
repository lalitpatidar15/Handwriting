import requests
import time
import json
import os

API_URL = "http://localhost:8000/api/v1/extract"
DOCS_URL = "http://localhost:8000/api/v1/documents"

def test_extraction(image_path: str):
    print(f"🚀 Testing DOC-INTEL Enterprise API with {image_path}...")
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image {image_path} not found.")
        return

    # Prepare the multipart form data
    with open(image_path, "rb") as img_file:
        files = {"file": (os.path.basename(image_path), img_file, "image/jpeg")}
        data = {
            "provider": "local",
            "force_mode": "",
            "high_fidelity": True
        }
        
        start_time = time.time()
        print("⏳ Sending request to API (this might take 10-30 seconds depending on OCR)...")
        
        try:
            response = requests.post(API_URL, files=files, data=data)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print("\n✅ API Response SUCCESS!")
                print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")
                print("\n📊 Extracted Data:")
                print(json.dumps(result, indent=2))
                
                # Check if it was saved to DB
                doc_id = result.get("document_id")
                if doc_id:
                    print(f"\n🔍 Verifying Database Record (ID: {doc_id})...")
                    db_response = requests.get(f"{DOCS_URL}/{doc_id}")
                    if db_response.status_code == 200:
                        print("✅ Record successfully retrieved from Database!")
                    else:
                        print(f"❌ Failed to retrieve from DB: {db_response.status_code}")
                
            else:
                print(f"\n❌ API Error ({response.status_code}):")
                print(response.text)
                
        except requests.exceptions.ConnectionError:
            print("\n❌ Connection Error: Is the FastAPI server running?")
            print("Run: uvicorn server:app --reload --port 8000")

if __name__ == "__main__":
    # Wait for the server to start if we just launched it
    print("Waiting 3 seconds for server to be ready...")
    time.sleep(3)
    
    # We will test with vanguard_debug.jpg which exists in the folder
    test_image = "vanguard_debug.jpg"
    test_extraction(test_image)
