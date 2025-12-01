import os
import json
import base64
import mimetypes
import io
import re
from pathlib import Path
from pdf2image import convert_from_path
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv
load_dotenv()
# 1. CONFIGURATION
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "allenai_olmOCR-2-7B-1025-Q5_K_M.gguf"

# 2. FOLDER SETTINGS
INPUT_FOLDER = "./images"
OUTPUT_FOLDER = "./output_model_olmocr"

# 3. PROMPT
BLOCK_START = "```json"
BLOCK_END = "```"

SYSTEM_PROMPT = """
You are an advanced medical OCR assistant. 
Analyze the image and return a JSON object strictly following this schema.
Wrap your response in a markdown code block like this:
""" + BLOCK_START + """
{
  ... your json here ...
}
""" + BLOCK_END + """

Schema:
{
  "generalInfo": [{"label": "string", "readText": {"value": "string"}}],
  "medicines": [{"slNo": int, "readText": {"medicine": "string", "dosage": "string", "timing": "string", "duration": "string"}}],
  "multivitamins": [{"slNo": int, "readText": {"medicine": "string", "dosage": "string", "timing": "string", "frequency": "string", "duration": "string"}}],
  "injections": [{"slNo": int, "readText": {"medicine": "string", "dosage": "string", "timing": "string", "frequency": "string", "duration": "string"}}],
  "radiologicalTests": [{"slNo": int, "readText": {"testName": "string", "additionalInfo": "string"}}],
  "pathologicalTests": [{"slNo": int, "readText": {"testName": "string", "additionalInfo": "string"}}]
}
Categorize items correctly (e.g., Iron/Calcium go to 'multivitamins', 'X-Ray' goes to 'radiologicalTests'). Return NULL for a key if NOT PRESENT.
Return ONLY JSON.
"""

def extract_json_from_response(response_text):
    """
    Parses the LLM response to find JSON data.
    1. Looks for ```json ... ``` blocks.
    2. Fallback: Looks for { ... } structure.
    3. Converts string to Python Dict.
    """
    if not response_text:
        return None

    try:
        # Pattern 1: Look for markdown code blocks
        match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Pattern 2: Look for raw JSON (starting with { and ending with })
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                # Fallback: Assume the whole text is JSON
                json_str = response_text

        # Clean up any potential leading/trailing whitespace
        return json.loads(json_str)

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        # Return the raw text wrapped in an object so data isn't lost
        return {"error": "parsing_failed", "raw_content": response_text}


def encode_image(image_input):
    """Encodes an image to base64 and determines the correct MIME type."""
    if isinstance(image_input, str):
        # Case 1: File Path
        mime_type, _ = mimetypes.guess_type(image_input)
        if not mime_type:
            mime_type = "image/jpeg"
        
        with open(image_input, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string, mime_type
    else:
        # Case 2: PIL Image
        buffered = io.BytesIO()
        if image_input.mode in ("RGBA", "P"):
            image_input = image_input.convert("RGB")
        image_input.save(buffered, format="JPEG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return encoded_string, "image/jpeg"

def call_model(client, base64_image, mime_type):
    """Sends the request to the local API."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return None

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    valid_exts = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'}
    files = [f for f in os.listdir(INPUT_FOLDER) if os.path.splitext(f)[1].lower() in valid_exts]
    
    print(f"Found {len(files)} documents. Starting processing...\n")

    for i, filename in enumerate(files):
        file_path = os.path.join(INPUT_FOLDER, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.json")
        
        if os.path.exists(output_path):
            print(f"Skipping {filename}")
            continue

        print(f"Processing [{i+1}/{len(files)}]: {filename}")
        
        doc_results = {"filename": filename, "pages": []}

        try:
            # === PDF HANDLING ===
            if filename.lower().endswith('.pdf'):
                print("   Converting PDF to images...")
                pages = convert_from_path(file_path, dpi=300)
                
                for page_idx, page_img in enumerate(pages):
                    print(f"   Page {page_idx + 1}/{len(pages)}...")
                    b64, mime = encode_image(page_img) 
                    raw_text = call_model(client, b64, mime)
                    
                    parsed_json = extract_json_from_response(raw_text)

                    doc_results["pages"].append({
                        "page_number": page_idx + 1,
                        "content": parsed_json
                    })
            
            # === IMAGE HANDLING ===
            else:
                print("   Reading Image...")
                b64, mime = encode_image(file_path) 
                raw_text = call_model(client, b64, mime)
                
                parsed_json = extract_json_from_response(raw_text)
                
                doc_results["pages"].append({
                    "page_number": 1,
                    "content": parsed_json
                })

            # Save the result
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(doc_results, f, indent=2, ensure_ascii=False)
            print(f"Saved to {output_path}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    main()