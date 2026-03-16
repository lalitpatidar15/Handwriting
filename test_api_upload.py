import requests
import os

url = "http://127.0.0.1:8000/api/v1/process_document"
# Create dummy file
with open("test_dummy.jpg", "wb") as f:
    f.write(b"dummy image data")

try:
    with open("test_dummy.jpg", "rb") as f:
        files = {"file": ("test_dummy.jpg", f, "image/jpeg")}
        data = {
            "cloud_mode": "false",
            "provider": "gemini",
            "high_fidelity": "true",
            "force_mode": ""
        }
        print("Sending POST request to:", url)
        response = requests.post(url, files=files, data=data, timeout=30)
        print("Status Code:", response.status_code)
        print("Response:", response.text)
except Exception as e:
    print("Error:", e)
finally:
    if os.path.exists("test_dummy.jpg"):
        os.remove("test_dummy.jpg")
