import runpod
import json
import urllib.request
import urllib.parse
import time
import os
import base64
import uuid
import websocket

COMFY_HOST = "127.0.0.1:8188"
COMFY_API_AVAILABLE_INTERVAL_MS = 50
COMFY_API_AVAILABLE_MAX_RETRIES = 500
COMFY_POLLING_INTERVAL_MS = int(os.environ.get("COMFY_POLLING_INTERVAL_MS", 250))
COMFY_POLLING_MAX_RETRIES = int(os.environ.get("COMFY_POLLING_MAX_RETRIES", 500))
COMFY_OUTPUT_PATH = os.environ.get("COMFY_OUTPUT_PATH", "/comfyui/output")


def wait_for_comfy():
    for _ in range(COMFY_API_AVAILABLE_MAX_RETRIES):
        try:
            urllib.request.urlopen(f"http://{COMFY_HOST}/system_stats")
            return True
        except Exception:
            time.sleep(COMFY_API_AVAILABLE_INTERVAL_MS / 1000)
    raise RuntimeError("ComfyUI did not become available in time")


def upload_images(images):
    """Upload images/audio files to ComfyUI's input directory."""
    if not images:
        return
    for item in images:
        name = item["name"]
        data = base64.b64decode(item["image"])
        form_data = urllib.parse.urlencode({}).encode()

        boundary = "----FormBoundary" + uuid.uuid4().hex
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + data + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            f"http://{COMFY_HOST}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        urllib.request.urlopen(req)


def queue_prompt(prompt, client_id):
    payload = json.dumps({"prompt": prompt, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"http://{COMFY_HOST}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    response = urllib.request.urlopen(req)
    return json.loads(response.read())


def get_history(prompt_id):
    with urllib.request.urlopen(f"http://{COMFY_HOST}/history/{prompt_id}") as r:
        return json.loads(r.read())


def wait_for_prompt(ws, prompt_id):
    """Wait for prompt to complete. Returns dict of node_id -> elapsed_seconds."""
    node_timings = {}
    current_node = None
    node_start = None
    try:
        for _ in range(COMFY_POLLING_MAX_RETRIES):
            out = ws.recv()
            if isinstance(out, str):
                msg = json.loads(out)
                if msg.get("type") == "executing":
                    data = msg.get("data", {})
                    if data.get("prompt_id") != prompt_id:
                        continue
                    node = data.get("node")
                    now = time.time()
                    # Record duration of previous node
                    if current_node and node_start:
                        node_timings[current_node] = round(now - node_start, 2)
                    if node is None:
                        # Prompt finished
                        break
                    current_node = node
                    node_start = now
            time.sleep(COMFY_POLLING_INTERVAL_MS / 1000)
    finally:
        ws.close()
    return node_timings


def collect_outputs(prompt_id):
    """
    Collect all image AND video outputs from ComfyUI history.
    Returns list of dicts: {"filename": str, "data": base64str, "type": "image"|"video"}
    """
    history = get_history(prompt_id)
    if prompt_id not in history:
        return []

    outputs = []
    for node_output in history[prompt_id]["outputs"].values():
        # Capture images (SaveImage / PreviewImage nodes)
        for img in node_output.get("images", []):
            file_path = os.path.join(
                COMFY_OUTPUT_PATH,
                img.get("subfolder", ""),
                img["filename"],
            )
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    outputs.append({
                        "filename": img["filename"],
                        "data": base64.b64encode(f.read()).decode("utf-8"),
                        "type": "image",
                    })

        # Capture videos (VHS_VideoCombine and similar nodes store under "gifs")
        for vid in node_output.get("gifs", []):
            file_path = os.path.join(
                COMFY_OUTPUT_PATH,
                vid.get("subfolder", ""),
                vid["filename"],
            )
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    outputs.append({
                        "filename": vid["filename"],
                        "data": base64.b64encode(f.read()).decode("utf-8"),
                        "type": "video",
                    })

    return outputs


def handler(job):
    job_input = job["input"]
    workflow = job_input.get("workflow")
    if not workflow:
        return {"error": "Missing 'workflow' in input"}

    images = job_input.get("images", [])

    wait_for_comfy()
    upload_images(images)

    client_id = str(uuid.uuid4())

    # Connect to WebSocket BEFORE submitting so we don't miss early node events
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")

    result = queue_prompt(workflow, client_id)
    prompt_id = result["prompt_id"]

    node_timings = wait_for_prompt(ws, prompt_id)

    outputs = collect_outputs(prompt_id)

    if not outputs:
        return {"status": "success_no_outputs", "images": []}

    # Separate images and videos for clarity
    videos = [o for o in outputs if o["type"] == "video"]
    imgs = [o for o in outputs if o["type"] == "image"]

    return {
        "status": "success",
        "images": imgs,
        "videos": videos,
        "node_timings": node_timings,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
