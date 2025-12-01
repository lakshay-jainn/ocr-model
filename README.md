# Medical OCR Pipeline

Local OCR server using `olmOCR` with `llama.cpp` via Docker.

## Setup

1. **Download models** (~6-7 GB):

   ```bash
   chmod +x setup_models.sh
   ./setup_models.sh
   ```

2. **Create `.env` file**:

   ```env
   API_KEY=your-secret-key
   ```

3. **Start the server**:

   ```bash
   docker-compose up -d
   ```

4. **Test** (optional):
   ```bash
   python test_ocr.py ./image.png
   ```

## Compare Model Results

I have already run inference using both **olmOCR** (local) and **Gemini** (cloud) on all images.

To compare:

1. View the source image in `images/`
2. Check olmOCR output in `output_model_olmocr/`
3. Check Gemini output in `output_model_gemini/`
