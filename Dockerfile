# Base image — ComfyUI + RunPod serverless handler pre-installed
FROM runpod/worker-comfyui:latest-base

# Update ComfyUI to latest (SeedVR2 requires V3 API)
RUN cd /comfyui && \
    git fetch origin && \
    git checkout origin/master

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

# —— Models ————————————————————————————————————————————————————————————————————
# SeedVR2 node auto-downloads models from HuggingFace on first run
# (~8.2GB DiT + 478MB VAE, takes ~20s at worker download speeds)
RUN mkdir -p /comfyui/models/SEEDVR2

# —— Custom handler with video output support ————————————————————————————————
COPY rp_handler.py /rp_handler.py
RUN handler=$(grep -rl "success_no_images" / --include="*.py" 2>/dev/null | head -1) && \
    echo "Found base handler at: $handler" && \
    cp /rp_handler.py "$handler"
