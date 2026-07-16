# infer_realtime.py
"""
Realtime inference script for sign language MVP.
Usage example:
  python infer_realtime.py --checkpoint checkpoints/best_model.pth --seq_len 32 --interval 0.5 --cam 0
"""

import argparse
import time
from collections import deque, Counter
import numpy as np
import cv2
import torch
import mediapipe as mp
import os

from models.pose_lstm import PoseLSTM

mp_holistic = mp.solutions.holistic

NUM_POSE = 33
NUM_HAND = 21
FEATURES_PER_LANDMARK = 3
FEATURE_DIM = (NUM_POSE + NUM_HAND + NUM_HAND) * FEATURES_PER_LANDMARK  # 225

def enhance_frame(frame):
    """Apply CLAHE contrast enhancement to improve landmark detection."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)



def extract_kps_from_results(results):
    landmark_list = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmark_list.append([lm.x, lm.y, lm.z])
    else:
        landmark_list += [[0.0, 0.0, 0.0]] * NUM_POSE

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmark_list.append([lm.x, lm.y, lm.z])
    else:
        landmark_list += [[0.0, 0.0, 0.0]] * NUM_HAND

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmark_list.append([lm.x, lm.y, lm.z])
    else:
        landmark_list += [[0.0, 0.0, 0.0]] * NUM_HAND

    return np.array(landmark_list).reshape(-1).astype(np.float32)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def smooth_prediction(pred_buffer, prob_buffer):
    votes = [p for p in pred_buffer if p is not None]
    if not votes:
        return None, 0.0
    cnt = Counter(votes)
    best = cnt.most_common(1)[0][0]
    frac = cnt[best] / len(votes)
    return best, frac


def build_seq_from_buffer(buf, seq_len):
    arr = list(buf)
    if len(arr) == 0:
        return np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)
    arr = np.stack(arr, axis=0)

    if arr.shape[0] >= seq_len:
        arr = arr[-seq_len:]
    else:
        pad_len = seq_len - arr.shape[0]
        pad = np.repeat(arr[-1:], pad_len, axis=0)
        arr = np.concatenate([arr, pad], axis=0)

    return arr.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--interval", type=float, default=0.6)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--conf_thresh", type=float, default=0.55)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ck = torch.load(args.checkpoint, map_location="cpu")
    labels_map = ck.get("labels")

    if isinstance(labels_map, dict):
        idx2label = {v: k for k, v in labels_map.items()}
        labels = [idx2label[i] for i in range(len(idx2label))]
    else:
        labels = list(labels_map)

    num_classes = len(labels)

    model = PoseLSTM(feature_dim=FEATURE_DIM, hidden=128, num_layers=2, num_classes=num_classes)
    model.load_state_dict(ck["model_state"])
    model.eval()
    model.to(torch.device("cpu"))

    frame_buf = deque(maxlen=args.seq_len)
    pred_buf = deque(maxlen=args.smooth_window)
    conf_buf = deque(maxlen=args.smooth_window)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.cam}")

    # ---------------- FULL SCREEN WINDOW HERE ----------------
    window_name = "Sign-Lang Translator (Press Q to Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 700)
    # ----------------------------------------------------------

    last_infer_t = 0.0

    with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            refine_face_landmarks=True) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = enhance_frame(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            kps = extract_kps_from_results(results)
            frame_buf.append(kps)

            now = time.time()
            if now - last_infer_t >= args.interval:
                last_infer_t = now
                seq = build_seq_from_buffer(frame_buf, args.seq_len)
                x = torch.tensor(seq).unsqueeze(0)

                with torch.no_grad():
                    logits = model(x)
                probs = softmax(logits.numpy().squeeze(0))
                top_idx = int(np.argmax(probs))
                top_conf = float(probs[top_idx])

                if top_conf >= args.conf_thresh:
                    pred_buf.append(top_idx)
                    conf_buf.append(top_conf)
                else:
                    pred_buf.append(None)
                    conf_buf.append(0.0)

                pred_idx, frac = smooth_prediction(pred_buf, conf_buf)
                if pred_idx is not None:
                    disp = f"{labels[pred_idx]} ({frac:.2f})"
                else:
                    disp = "..."

            cv2.rectangle(frame, (0, frame.shape[0] - 40),
                          (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)

            if 'disp' in locals():
                cv2.putText(frame, disp, (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
