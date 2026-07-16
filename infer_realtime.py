import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque, Counter
from models.pose_lstm import PoseLSTM

mp_holistic = mp.solutions.holistic

NUM_POSE, NUM_HAND = 33, 21
FEATURE_DIM = (NUM_POSE + NUM_HAND + NUM_HAND) * 3  # 225
SEQ_LEN     = 32
CONF_THRESH = 0.55
SMOOTH_WIN  = 5


# ── Helpers (copied exactly from your infer_realtime.py) ───────────────────
def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def extract_kps(results):
    lms = []
    for src, n in [(results.pose_landmarks,       NUM_POSE),
                   (results.left_hand_landmarks,  NUM_HAND),
                   (results.right_hand_landmarks, NUM_HAND)]:
        if src:
            lms += [[lm.x, lm.y, lm.z] for lm in src.landmark]
        else:
            lms += [[0.0, 0.0, 0.0]] * n
    return np.array(lms).reshape(-1).astype(np.float32)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def build_seq(buf):
    arr = np.stack(list(buf), axis=0)
    if arr.shape[0] >= SEQ_LEN:
        return arr[-SEQ_LEN:]
    pad = np.repeat(arr[-1:], SEQ_LEN - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0).astype(np.float32)

def smooth_prediction(pred_buf):
    votes = [p for p in pred_buf if p is not None]
    if not votes:
        return None, 0.0
    cnt  = Counter(votes)
    best = cnt.most_common(1)[0][0]
    return best, cnt[best] / len(votes)


# ── InferenceEngine class ──────────────────────────────────────────────────
class InferenceEngine:
    def __init__(self, checkpoint_path: str):
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ck = torch.load(checkpoint_path, map_location="cpu")
        lmap = ck["labels"]
        if isinstance(lmap, dict):
            idx2label  = {v: k for k, v in lmap.items()}
            self.labels = [idx2label[i] for i in range(len(idx2label))]
        else:
            self.labels = list(lmap)

        self.model = PoseLSTM(
            feature_dim=FEATURE_DIM, hidden=128,
            num_layers=2, num_classes=len(self.labels)
        )
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()

        # Shared MediaPipe holistic instance
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            refine_face_landmarks=False,
        )

        # Per-session state: {session_id: {"frame_buf", "pred_buf", "no_hands"}}
        self._sessions: dict = {}

    def _get_session(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "frame_buf": deque(maxlen=SEQ_LEN),
                "pred_buf":  deque(maxlen=SMOOTH_WIN),
                "no_hands":  0,
            }
        return self._sessions[session_id]

    def predict_frame(self, frame: np.ndarray, session_id: str = "default"):
        """
        Accept a BGR numpy frame from the phone,
        run the exact same pipeline as infer_realtime.py,
        return (label_str | None, confidence_float).
        """
        sess = self._get_session(session_id)

        frame     = enhance_frame(frame)
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = self.holistic.process(rgb)

        hands_visible = (results.left_hand_landmarks  is not None or
                         results.right_hand_landmarks is not None)

        # Reset smoothing buffer if hands disappear (mirrors your no_hands_frames > 10)
        if not hands_visible:
            sess["no_hands"] += 1
            if sess["no_hands"] > 10:
                sess["pred_buf"].clear()
                sess["frame_buf"].clear()
            return None, 0.0
        else:
            sess["no_hands"] = 0

        kps = extract_kps(results)
        sess["frame_buf"].append(kps)

        if len(sess["frame_buf"]) < 4:
            return None, 0.0

        # LSTM inference
        seq = build_seq(sess["frame_buf"])
        x   = torch.tensor(seq).unsqueeze(0)
        with torch.no_grad():
            probs = softmax(self.model(x).cpu().numpy().squeeze(0))

        top_idx  = int(np.argmax(probs))
        top_conf = float(probs[top_idx])

        # Smoothing — same logic as your smooth_prediction()
        if top_conf >= CONF_THRESH:
            sess["pred_buf"].append(top_idx)
        else:
            sess["pred_buf"].append(None)

        pred_idx, vote_frac = smooth_prediction(sess["pred_buf"])
        if pred_idx is not None:
            return self.labels[pred_idx], vote_frac
        return None, top_conf

    def close(self):
        self.holistic.close()