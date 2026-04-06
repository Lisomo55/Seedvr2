# Base image — ComfyUI + RunPod serverless handler pre-installed
FROM runpod/worker-comfyui:latest-base

# Pin to exact ComfyUI version
RUN cd /comfyui && \
    git fetch origin && \
    git checkout ed7c2c65

# —— Custom Nodes ——————————————————————————————————————————————————————————————

# SeedVR2 Video Upscaler
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler && \
    cd ComfyUI-SeedVR2_VideoUpscaler && \
    pip install -r requirements.txt --break-system-packages

# VideoHelperSuite (for video loading/saving)
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt --break-system-packages

# —— Pre-download Models ——————————————————————————————————————————————————————

# Install wget (not in base image)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# SeedVR2 7B sharp fp8 — sharpest fp8 variant, fits on 24-32GB GPUs
RUN mkdir -p /comfyui/models/SEEDVR2 && \
    wget --no-verbose --content-disposition \
        "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors" \
        -O /comfyui/models/SEEDVR2/seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors

# VAE (required for all configurations)
RUN wget --no-verbose --content-disposition \
        "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/ema_vae_fp16.safetensors" \
        -O /comfyui/models/SEEDVR2/ema_vae_fp16.safetensors

# —— Custom handler with video output support ————————————————————————————————
COPY rp_handler.py /rp_handler.py
RUN handler=$(grep -rl "success_no_images" / --include="*.py" 2>/dev/null | head -1) && \
    echo "Found base handler at: $handler" && \
    cp /rp_handler.py "$handler"
