import os
import base64
import sys
import mimetypes
from openai import OpenAI #pip install openai
from dotenv import load_dotenv #pip install python-dotenv

load_dotenv()
# Configuration
BASE_URL = "http://localhost:8000/v1"
API_KEY = os.getenv("API_KEY")
print(API_KEY)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_connection(client):
    print(f"Testing connection to {BASE_URL}...")
    try:
        client.models.list()
        print("Success: Connected to server.")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

def test_ocr(client, image_path):
    print(f"Processing '{image_path}'...")
    
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return
     # Determine the correct MIME type (e.g., image/png, image/jpeg)
    mime_type, _ = mimetypes.guess_type(image_path)
    print(mime_type)
    if not mime_type:
        mime_type = "image/jpeg" # Fallback default

    try:
        response = client.chat.completions.create(
            model="allenai_olmOCR-2-7B-1025-Q5_K_M.gguf",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Convert this image to markdown."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encode_image(image_path)}"}}
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=4096
        )
        print("\nOCR Result:\n")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    if test_connection(client):
        if len(sys.argv) > 1:
            test_ocr(client, sys.argv[1])
        else:
            print(f"Usage: python {sys.argv[0]} path/to/document.jpg")