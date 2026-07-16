#!/usr/bin/env python3
"""
record_sign.py

Usage:
    python record_sign.py <label> <count> [--dur 3] [--fps 30] [--w 640] [--h 480] [--auto]

Examples:
    python record_sign.py hello 20
    python record_sign.py thanks 15 --dur 2 --auto
"""

import cv2
import os
import argparse
import time
from datetime import datetime

RAW_DIR = "data/raw_videos"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def next_available_index(label):
    """Find next index for label files existing in RAW_DIR."""
    existing = [f for f in os.listdir(RAW_DIR) if f.startswith(label + "_") and f.endswith(".mp4")]
    indices = []
    for f in existing:
        try:
            idx = int(f.split("_")[-1].split(".")[0])
            indices.append(idx)
        except:
            pass
    return max(indices) + 1 if indices else 1

def record_clip(save_path, duration=3, fps=30, width=640, height=480, cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index {}).".format(cam_index))

    # Try to set resolution (may be ignored by some webcams)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    print(f"Recording -> {save_path}  (duration: {duration}s, fps: {fps})")
    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: empty frame from camera.")
                break
            out.write(frame)
            # show preview
            cv2.imshow("Recording (press 'q' to cancel)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Recording cancelled by user.")
                break
            if time.time() - start_time >= duration:
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def countdown(seconds):
    for s in range(seconds, 0, -1):
        print(f"Starting in {s}...", end="\r")
        time.sleep(1)
    print(" " * 30, end="\r")  # clear line

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="Label name for this sign (e.g., hello, thanks)")
    parser.add_argument("count", type=int, help="How many clips to record for this label")
    parser.add_argument("--dur", type=float, default=3.0, help="Duration (seconds) for each clip")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second to record")
    parser.add_argument("--w", type=int, default=640, help="Width of recorded video")
    parser.add_argument("--h", type=int, default=480, help="Height of recorded video")
    parser.add_argument("--auto", action="store_true", help="Auto-record samples one after another (no manual start)")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (0,1,...)")
    args = parser.parse_args()

    ensure_dir(RAW_DIR)
    start_idx = next_available_index(args.label)

    print(f"\nPreparing to record {args.count} samples of '{args.label}'.")
    print("Make sure the camera is pointing at you, lighting is good, and your upper body is visible.")
    print("You can press 'q' during recording to cancel a clip.\n")
    time.sleep(1.0)

    for i in range(start_idx, start_idx + args.count):
        filename = f"{args.label}_{i:02d}.mp4"
        save_path = os.path.join(RAW_DIR, filename)

        if args.auto:
            # small 1-second notice before auto-record
            print(f"Auto recording sample {i - start_idx + 1}/{args.count} -> {filename}")
            countdown(1)
            try:
                record_clip(save_path, duration=args.dur, fps=args.fps, width=args.w, height=args.h, cam_index=args.cam)
                print(f"Saved: {save_path}\n")
            except Exception as e:
                print("Error recording:", e)
                break
            time.sleep(0.6)  # short gap
        else:
            print(f"Ready to record sample {i - start_idx + 1}/{args.count} -> {filename}")
            print("Press ENTER to start recording this sample, or type 's' then ENTER to skip, or 'q' to quit.")
            user = input("> ").strip().lower()
            if user == "q":
                print("Exiting recording.")
                break
            if user == "s":
                print("Skipping this sample.\n")
                continue

            # countdown and record
            countdown(3)
            try:
                record_clip(save_path, duration=args.dur, fps=args.fps, width=args.w, height=args.h, cam_index=args.cam)
                print(f"Saved: {save_path}\n")
            except Exception as e:
                print("Error recording:", e)
                break

    print("Recording session finished.")

if __name__ == "__main__":
    main()
