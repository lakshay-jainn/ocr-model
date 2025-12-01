#!/bin/bash

# --- Configuration ---
MODEL_DIR="$(pwd)/ocr-models"
HF_REPO="https://huggingface.co/bartowski/allenai_olmOCR-2-7B-1025-GGUF/resolve/main"

MODEL_FILE="allenai_olmOCR-2-7B-1025-Q5_K_M.gguf"
PROJ_FILE="mmproj-allenai_olmOCR-2-7B-1025-f16.gguf"
TEMPLATE_FILE="olm_template.j2"

# --- Setup Directories ---
set -e
if [ ! -d "$MODEL_DIR" ]; then 
    echo "Creating directory $MODEL_DIR..."
    mkdir -p "$MODEL_DIR"
fi

# 1. Create Template
if [ ! -f "$MODEL_DIR/$TEMPLATE_FILE" ]; then
    echo "Creating custom prompt template..."
    cat <<'EOF' > "$MODEL_DIR/$TEMPLATE_FILE"
{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}<|im_end|>
{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}
EOF
    echo "Template created."
fi

# 2. Download Files
echo "Checking for models..."
wget -c -P "$MODEL_DIR" "$HF_REPO/$MODEL_FILE"
wget -c -P "$MODEL_DIR" "$HF_REPO/$PROJ_FILE"

echo "Setup complete! You can now run: docker-compose up"