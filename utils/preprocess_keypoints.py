import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import mediapipe as mp

mp_holistic = mp.solutions.holistic

NUM_POSE = 33
NUM_HAND = 21
NUM_LANDMARKS = NUM_POSE + NUM_HAND + NUM_HAND
FEATURES_PER_LANDMARK = 3
FEATURE_DIM = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 225

def extract_kps_from_results(results):
    landmark_list = []
    # Pose
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmark_list.append([lm.x, lm.y, lm.z])
    else:
        landmark_list += [[0.0, 0.0, 0.0]] * NUM_POSE
    # Left hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmark_list.append([lm.x, lm.y, lm.z])
    else:
        landmark_list += [[0.0, 0.0, 0.0]] * NUM_HAND
    # Right hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmark_list.append([lm.x, lm.y, lm.z])
    else:
        landmark_list += [[0.0, 0.0, 0.0]] * NUM_HAND

    arr = np.array(landmark_list).reshape(-1)  # flattened (225)
    return arr

def enhance_frame(frame):
    """Apply CLAHE contrast enhancement to improve landmark detection."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def process_video(video_path, seq_len=32):
    cap = cv2.VideoCapture(video_path)
    frames = []
    lhand_missing = 0
    rhand_missing = 0
    total_frames = 0
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        refine_face_landmarks=False
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = enhance_frame(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            if not results.left_hand_landmarks:
                lhand_missing += 1
            if not results.right_hand_landmarks:
                rhand_missing += 1
            total_frames += 1
            kps = extract_kps_from_results(results)
            frames.append(kps)
    cap.release()
    if total_frames > 0:
        print(f"    {os.path.basename(video_path)}: LHand missing {lhand_missing}/{total_frames}, RHand missing {rhand_missing}/{total_frames}")

    if len(frames) == 0:
        return np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)

    arr = np.stack(frames, axis=0)

    if arr.shape[0] == seq_len:
        seq = arr
    elif arr.shape[0] > seq_len:
        idxs = np.linspace(0, arr.shape[0] - 1, seq_len).astype(int)
        seq = arr[idxs]
    else:
        pad_len = seq_len - arr.shape[0]
        pad = np.repeat(arr[-1:], pad_len, axis=0)
        seq = np.concatenate([arr, pad], axis=0)

    return seq.astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/raw_videos")
    parser.add_argument("--out_dir", type=str, default="data/keypoints")
    parser.add_argument("--seq_len", type=int, default=32)
    args = parser.parse_args()

    inp = args.input_dir
    out = args.out_dir

    os.makedirs(out, exist_ok=True)

    videos = [f for f in os.listdir(inp) if f.lower().endswith(".mp4")]
    print(f"Found {len(videos)} videos.")

    for v in tqdm(videos):
        in_path = os.path.join(inp, v)
        base = v.replace(".mp4", "")
        out_path = os.path.join(out, base + ".npy")

        seq = process_video(in_path, seq_len=args.seq_len)
        np.save(out_path, seq)

    print("DONE!")

if __name__ == "__main__":
    main()
