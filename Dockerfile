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

# —— Models via Network Volume ————————————————————————————————————————————————
# SeedVR2 models dir points to network volume to avoid container disk limits
# Upload models to /runpod-volume/models/SEEDVR2/ on the network volume
RUN rm -rf /comfyui/models/SEEDVR2 && \
    ln -sf /runpod-volume/models/SEEDVR2 /comfyui/models/SEEDVR2

# —— Custom handler with video output support ————————————————————————————————
COPY rp_handler.py /rp_handler.py
RUN handler=$(grep -rl "success_no_images" / --include="*.py" 2>/dev/null | head -1) && \
    echo "Found base handler at: $handler" && \
    cp /rp_handler.py "$handler"
