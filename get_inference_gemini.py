import os
import json
import re
import mimetypes
import google.generativeai as genai #pip install google-generativeai
from dotenv import load_dotenv
load_dotenv()
# 1. CONFIGURATION
API_KEY = os.getenv("GEMINI_API_KEY")  # Replace with actual key

MODEL_NAME = "gemini-2.5-flash" 

# 2. FOLDER SETTINGS
INPUT_FOLDER = "./images"
OUTPUT_FOLDER = "./output_model_gemini"

# 3. PROMPT CONFIGURATION
SYSTEM_PROMPT = """
You are an advanced medical OCR assistant. 
Analyze the ENTIRE document provided and return a JSON object strictly following this schema.

Schema Rules:
1. Categorize items correctly (e.g., Iron/Calcium go to 'multivitamins').
2. Return NULL for a key if NOT PRESENT.
3. If the document has multiple pages, extract data from ALL pages into the lists below.

Output Schema:
{
  "generalInfo": [{"label": "string", "readText": {"value": "string"}}],
  "medicines": [{"slNo": int, "readText": {"medicine": "string", "dosage": "string", "timing": "string", "duration": "string"}}],
  "multivitamins": [{"slNo": int, "readText": {"medicine": "string", "dosage": "string", "timing": "string", "frequency": "string", "duration": "string"}}],
  "injections": [{"slNo": int, "readText": {"medicine": "string", "dosage": "string", "timing": "string", "frequency": "string", "duration": "string"}}],
  "radiologicalTests": [{"slNo": int, "readText": {"testName": "string", "additionalInfo": "string"}}],
  "pathologicalTests": [{"slNo": int, "readText": {"testName": "string", "additionalInfo": "string"}}]
}
"""

def configure_genai():
    genai.configure(api_key=API_KEY)

def extract_json_from_response(response_text):
    """Parses the LLM response to find JSON data."""
    if not response_text:
        return None

    try:
        # Strip markdown code blocks if present
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except json.JSONDecodeError:
        # Fallback regex search
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        return {"error": "parsing_failed", "raw_content": response_text}

def process_file(model, file_path, mime_type):
    """Reads file bytes and sends them directly to Gemini."""
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # Construct the content part with inline data
        content_parts = [
            SYSTEM_PROMPT,
            {
                "mime_type": mime_type,
                "data": file_bytes
            }
        ]

        response = model.generate_content(
            content_parts,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        return response.text
    except Exception as e:
        print(f"API Error processing {file_path}: {e}")
        return None

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    configure_genai()
    model = genai.GenerativeModel(MODEL_NAME)
    
    # Supported extensions
    valid_exts = {'.pdf', '.png', '.jpg', '.jpeg', '.webp', '.heic'}
    files = [f for f in os.listdir(INPUT_FOLDER) if os.path.splitext(f)[1].lower() in valid_exts]
    
    print(f"Found {len(files)} documents. Starting native processing...\n")

    for i, filename in enumerate(files):
        file_path = os.path.join(INPUT_FOLDER, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.json")
        
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            # Fallback for common types if OS registry is missing them
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.pdf': mime_type = 'application/pdf'
            elif ext in ['.jpg', '.jpeg']: mime_type = 'image/jpeg'
            elif ext == '.png': mime_type = 'image/png'
            else: mime_type = 'application/octet-stream'

        if os.path.exists(output_path):
            print(f"Skipping {filename} (Already processed)")
            continue

        print(f"Processing [{i+1}/{len(files)}]: {filename} ({mime_type})")
        
        # Call Gemini (Handles PDF or Image natively)
        raw_text = process_file(model, file_path, mime_type)
        parsed_json = extract_json_from_response(raw_text)

        # Save result
        result_wrapper = {
            "filename": filename,
            "mime_type": mime_type,
            "content": parsed_json
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_wrapper, f, indent=2, ensure_ascii=False)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()