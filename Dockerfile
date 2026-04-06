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

# —— Models via Network Volume ————————————————————————————————————————————————
# Models are stored on a RunPod network volume mounted at /runpod-volume/
# Symlink so ComfyUI finds them at the expected path
RUN mkdir -p /comfyui/models/SEEDVR2 && \
    ln -sf /runpod-volume/models/SEEDVR2/seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors /comfyui/models/SEEDVR2/ && \
    ln -sf /runpod-volume/models/SEEDVR2/ema_vae_fp16.safetensors /comfyui/models/SEEDVR2/

# —— Custom handler with video output support ————————————————————————————————
COPY rp_handler.py /rp_handler.py
RUN handler=$(grep -rl "success_no_images" / --include="*.py" 2>/dev/null | head -1) && \
    echo "Found base handler at: $handler" && \
    cp /rp_handler.py "$handler"
