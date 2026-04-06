#!/usr/bin/env python3
"""
Upscale all shot videos using SeedVR2 on RunPod Serverless.
Reads from projects/current/video/, outputs to projects/current/video_upscaled/.
Submits all jobs at once to avoid cold starts.

Usage:
  python run_upscale.py
  python run_upscale.py --resolution 1080 --batch-size 5

Requirements:
  pip install requests python-dotenv
"""

import argparse
import base64
import glob
import json
import os
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

# --- Config ---
API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}" if ENDPOINT_ID else None
POLL_INTERVAL = 10
MAX_WAIT_SEC = 3600

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "projects", "current")
VIDEO_DIR = os.path.join(PROJECT_DIR, "video")
UPSCALED_DIR = os.path.join(PROJECT_DIR, "video_upscaled")

# SeedVR2 model filenames
DIT_MODEL = "seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors"
VAE_MODEL = "ema_vae_fp16.safetensors"


def get_headers():
    if not API_KEY:
        print("Error: Set RUNPOD_API_KEY in .env", file=sys.stderr)
        sys.exit(1)
    if not ENDPOINT_ID or ENDPOINT_ID == "your_seedvr2_endpoint_id_here":
        print("Error: Set RUNPOD_ENDPOINT_ID in .env", file=sys.stderr)
        sys.exit(1)
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def encode_file_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_seedvr2_workflow(
    video_name: str,
    resolution: int = 1080,
    batch_size: int = 5,
    color_correction: str = "lab",
    seed: int = 42,
) -> dict:
    """Build a ComfyUI workflow for SeedVR2 upscaling."""
    workflow = {
        # Load DiT model
        "1": {
            "inputs": {
                "model_name": DIT_MODEL,
                "device": "cuda",
                "enable_blockswap": False,
                "blockswap_double": 0,
                "blockswap_single": 0,
            },
            "class_type": "SeedVR2 (Down)Load DiT Model",
        },
        # Load VAE
        "2": {
            "inputs": {
                "vae_name": VAE_MODEL,
                "vae_tile_size": 256,
                "vae_tile_stride": 192,
            },
            "class_type": "SeedVR2 (Down)Load VAE Model",
        },
        # Load input video
        "3": {
            "inputs": {
                "video": video_name,
                "force_rate": 25,
                "force_size": "Disabled",
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
            "class_type": "VHS_LoadVideo",
        },
        # Upscale
        "4": {
            "inputs": {
                "resolution": resolution,
                "batch_size": batch_size,
                "color_correction": color_correction,
                "seed": seed,
                "dit_model": ["1", 0],
                "vae_model": ["2", 0],
                "images": ["3", 0],
            },
            "class_type": "SeedVR2 Video Upscaler",
        },
        # Save video
        "5": {
            "inputs": {
                "frame_rate": 25,
                "loop_count": 0,
                "filename_prefix": "SeedVR2_upscaled",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 17,
                "save_metadata": False,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
                "images": ["4", 0],
                "audio": ["3", 1],
            },
            "class_type": "VHS_VideoCombine",
        },
    }
    return workflow


def submit_job(payload: dict) -> str:
    headers = get_headers()
    resp = requests.post(f"{BASE_URL}/run", headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        print(f"  ERROR: Submission failed ({resp.status_code}): {resp.text[:500]}")
        return None
    data = resp.json()
    return data.get("id")


def poll_all_jobs(jobs: dict):
    headers = get_headers()
    pending = dict(jobs)
    results = {}
    start = time.time()
    seen_logs = {name: set() for name in jobs}

    while pending and (time.time() - start) < MAX_WAIT_SEC:
        for name in list(pending.keys()):
            job_id = pending[name]["job_id"]
            elapsed = int(time.time() - start)

            try:
                resp = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers, timeout=30)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                status = data.get("status", "UNKNOWN")

                logs = data.get("logs", "").strip()
                if logs:
                    for line in logs.splitlines():
                        if line and line not in seen_logs[name]:
                            seen_logs[name].add(line)
                            print(f"  [{elapsed:4d}s] {name}: {line}")

                if status == "COMPLETED":
                    print(f"  [{elapsed:4d}s] {name}: COMPLETED")
                    results[name] = {"status": "COMPLETED", "output": data.get("output", {})}
                    del pending[name]
                elif status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                    error = data.get("error", "Unknown error")
                    print(f"  [{elapsed:4d}s] {name}: {status} - {error}")
                    results[name] = {"status": status, "error": error}
                    del pending[name]
                else:
                    print(f"  [{elapsed:4d}s] {name}: {status}")

            except Exception as e:
                print(f"  [{elapsed:4d}s] {name}: poll error - {e}")

        if pending:
            time.sleep(POLL_INTERVAL)

    for name in pending:
        results[name] = {"status": "TIMEOUT"}
        print(f"  {name}: TIMEOUT")

    return results


def main():
    parser = argparse.ArgumentParser(description="Upscale shot videos with SeedVR2.")
    parser.add_argument("--resolution", type=int, default=1080, help="Target shortest edge (default: 1080)")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size, must follow 4n+1 (1,5,9,13...) (default: 5)")
    parser.add_argument("--color-correction", default="lab", help="Color correction method (default: lab)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not os.path.isdir(VIDEO_DIR):
        print(f"Error: Video directory not found: {VIDEO_DIR}", file=sys.stderr)
        sys.exit(1)

    video_files = sorted(
        [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")],
        key=lambda f: int("".join(c for c in f.split("_")[-1].split(".")[0] if c.isdigit()) or 0),
    )

    if not video_files:
        print(f"Error: No .mp4 files in {VIDEO_DIR}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(UPSCALED_DIR, exist_ok=True)

    print(f"Videos    : {len(video_files)}")
    print(f"Resolution: {args.resolution}p")
    print(f"Model     : {DIT_MODEL}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Submit all jobs at once
    jobs = {}
    for video_file in video_files:
        shot_name = video_file.replace(".mp4", "")
        video_path = os.path.join(VIDEO_DIR, video_file)

        workflow = build_seedvr2_workflow(
            video_name=video_file,
            resolution=args.resolution,
            batch_size=args.batch_size,
            color_correction=args.color_correction,
            seed=args.seed,
        )

        payload = {
            "input": {
                "workflow": workflow,
                "images": [
                    {"name": video_file, "image": encode_file_base64(video_path)},
                ],
            }
        }

        file_size = os.path.getsize(video_path)
        print(f"  {shot_name}: {file_size / 1024:.0f} KB")
        job_id = submit_job(payload)

        if job_id:
            print(f"  {shot_name}: submitted (job {job_id})")
            jobs[shot_name] = {
                "job_id": job_id,
                "output_path": os.path.join(UPSCALED_DIR, f"{shot_name}.mp4"),
            }
        else:
            print(f"  {shot_name}: FAILED to submit")

    if not jobs:
        print("No jobs submitted.", file=sys.stderr)
        sys.exit(1)

    print(f"\nAll {len(jobs)} jobs submitted. Polling...\n")

    results = poll_all_jobs(jobs)

    # Save completed videos
    success = 0
    failed = 0
    for shot_name, result in sorted(results.items()):
        if result["status"] == "COMPLETED":
            output = result["output"]
            videos = output.get("videos", [])
            if videos:
                video_data = videos[-1]["data"]
                output_path = jobs[shot_name]["output_path"]
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(video_data))
                print(f"  Saved: {output_path}")
                success += 1

                timings = output.get("node_timings", {})
                if timings:
                    total_time = sum(timings.values())
                    print(f"    Processing time: {total_time:.1f}s")
            else:
                print(f"  {shot_name}: no video in output")
                failed += 1
        else:
            failed += 1

    print(f"\nDone. {success}/{len(results)} videos upscaled to {UPSCALED_DIR}")
    if failed:
        print(f"{failed} jobs failed.")


if __name__ == "__main__":
    main()
