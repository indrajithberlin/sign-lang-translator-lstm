import time
import logging
from collections import deque, Counter
import numpy as np
import cv2
import torch
import mediapipe as mp
import threading
import platform

from models.pose_lstm import PoseLSTM

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

mp_holistic = mp.solutions.holistic

NUM_POSE = 33
NUM_HAND = 21
FEATURES_PER_LANDMARK = 3
FEATURE_DIM = (NUM_POSE + NUM_HAND + NUM_HAND) * FEATURES_PER_LANDMARK  # 225


def enhance_frame(frame):
    """Apply CLAHE contrast enhancement."""
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


def _open_camera(cam_index, width, height):
    """
    Open the camera using the correct backend for the current OS.
    CAP_DSHOW is Windows-only; other platforms use the default backend.
    """
    # FIX 1: Only use CAP_DSHOW on Windows — it silently fails on Linux/Mac
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        log.error(
            f"Camera index {cam_index} could not be opened. "
            "Try a different index (0, 1, 2...) or check permissions."
        )
    else:
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"Camera {cam_index} opened at {actual_w}x{actual_h}")

    return cap


class InferenceEngine:
    def __init__(
        self,
        checkpoint_path="checkpoints/best_model.pth",
        seq_len=32,
        interval=0.6,
        cam_index=0,
        width=640,
        height=480,
    ):
        self.seq_len = seq_len
        self.interval = interval
        self.cam_index = cam_index
        self.width = width
        self.height = height

        self.conf_thresh = 0.55
        self.smooth_window = 5

        # Load checkpoint
        ck = torch.load(checkpoint_path, map_location="cpu")
        labels_map = ck.get("labels")
        if isinstance(labels_map, dict):
            idx2label = {v: k for k, v in labels_map.items()}
            self.labels = [idx2label[i] for i in range(len(idx2label))]
        else:
            self.labels = list(labels_map)

        num_classes = len(self.labels)
        log.info(f"Loaded {num_classes} classes: {self.labels}")

        self.model = PoseLSTM(
            feature_dim=FEATURE_DIM, hidden=128, num_layers=2, num_classes=num_classes
        )
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()
        self.model.to(torch.device("cpu"))

        # Buffers
        self.frame_buf = deque(maxlen=self.seq_len)
        self.pred_buf = deque(maxlen=self.smooth_window)
        self.conf_buf = deque(maxlen=self.smooth_window)

        # Threading state
        self.cap = None
        self.holistic = None
        self.running = False
        self.thread = None
        self.camera_ready = False

        # Shared outputs — always access under self.lock
        self.current_frame = None  # JPEG bytes
        self.current_prediction = "..."
        self.current_confidence = 0.0
        self.lock = threading.Lock()

    def start(self):
        # FIX 1: cross-platform camera open
        self.cap = _open_camera(self.cam_index, self.width, self.height)
        self.camera_ready = self.cap.isOpened()

        # FIX 2: use model_complexity=1 (not 2) in the background thread.
        # model_complexity=2 is the slowest setting; it causes the loop to
        # fall so far behind that current_frame never gets populated in time
        # for the MJPEG stream, making the feed appear broken.
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            refine_face_landmarks=False,  # also saves time in the tight loop
        )

        self.web_holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            refine_face_landmarks=False,
        )

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        log.info("InferenceEngine started.")

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self.holistic is not None:
            self.holistic.close()
        if self.web_holistic is not None:
            self.web_holistic.close()
        if self.cap is not None:
            self.cap.release()
        log.info("InferenceEngine stopped.")

    def _run_loop(self):
        last_infer_t = 0.0
        no_hands_frames = 0

        while self.running:
            read_ok, frame = self.cap.read()
            if not read_ok:
                log.warning("Camera read failed — retrying...")
                time.sleep(0.05)
                continue

            frame_enhanced = enhance_frame(frame)
            frame_rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(frame_rgb)

            # Hand presence tracking
            if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
                no_hands_frames += 1
            else:
                no_hands_frames = 0

            # Draw landmarks on the original (non-enhanced) frame for natural colour
            if results.face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    mp.solutions.holistic.FACEMESH_CONTOURS,
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(80, 110, 10), thickness=1, circle_radius=1
                    ),
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(80, 256, 121), thickness=1, circle_radius=1
                    ),
                )
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(80, 110, 10), thickness=2, circle_radius=2
                ),
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(80, 256, 121), thickness=2, circle_radius=2
                ),
            )
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )

            # Keypoint extraction
            kps = extract_kps_from_results(results)
            self.frame_buf.append(kps)

            # Inference at the configured interval
            now = time.time()
            if now - last_infer_t >= self.interval:
                last_infer_t = now
                seq = build_seq_from_buffer(self.frame_buf, self.seq_len)
                x = torch.tensor(seq).unsqueeze(0)

                with torch.no_grad():
                    logits = self.model(x)

                probs = softmax(logits.numpy().squeeze(0))
                top_idx = int(np.argmax(probs))
                top_conf = float(probs[top_idx])

                if top_conf >= self.conf_thresh:
                    self.pred_buf.append(top_idx)
                    self.conf_buf.append(top_conf)
                else:
                    self.pred_buf.append(None)
                    self.conf_buf.append(0.0)

                pred_idx, frac = smooth_prediction(self.pred_buf, self.conf_buf)

                with self.lock:
                    if pred_idx is not None:
                        self.current_prediction = self.labels[pred_idx]
                        self.current_confidence = frac
                    else:
                        self.current_prediction = "Waiting for sign..."
                        self.current_confidence = 0.0

            if no_hands_frames > 10:
                with self.lock:
                    self.current_prediction = "Waiting for sign..."
                    self.current_confidence = 0.0
                self.pred_buf.clear()
                self.conf_buf.clear()

            # FIX 3: read disp inside the lock to avoid a race condition
            # where current_prediction is updated mid-read
            with self.lock:
                pred_text = self.current_prediction
                pred_conf = self.current_confidence

            if pred_text != "Waiting for sign...":
                disp = f"{pred_text} ({pred_conf:.2f})"
            else:
                disp = pred_text

            # Overlay text
            if results.pose_landmarks:
                nose = results.pose_landmarks.landmark[0]
                h, w, _ = frame.shape
                cx, cy = int(nose.x * w), int(nose.y * h)
                (tw, th), _ = cv2.getTextSize(disp, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                tx = max(10, min(cx + 50, w - tw - 10))
                ty = max(th + 10, min(cy - 80, h - 10))
                cv2.rectangle(
                    frame, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10), (0, 0, 0), -1
                )
                cv2.putText(
                    frame, disp, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                )
            else:
                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
                cv2.putText(
                    frame, disp, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                )

            # FIX 4: use a separate variable (encode_ok) so 'read_ok' from
            # cap.read() is not accidentally overwritten before next iteration
            encode_ok, buffer = cv2.imencode(".jpg", frame)
            if encode_ok:
                with self.lock:
                    self.current_frame = buffer.tobytes()

            time.sleep(0.01)

    def get_latest_data(self):
        with self.lock:
            return self.current_frame, self.current_prediction, self.current_confidence
        
def predict_frame(self, frame: np.ndarray, session_id: str = "default"):
    """
    Accept a BGR numpy frame from the phone browser,
    run the same pipeline as _run_loop but without the camera.
    Returns (label_str | None, confidence_float).
    """
    # Per-session buffers (separate from the PC camera loop buffers)
    if not hasattr(self, '_session_bufs'):
        self._session_bufs = {}

    if session_id not in self._session_bufs:
        self._session_bufs[session_id] = {
            "frame_buf": deque(maxlen=self.seq_len),
            "pred_buf":  deque(maxlen=self.smooth_window),
            "conf_buf":  deque(maxlen=self.smooth_window),
            "no_hands":  0,
        }

    sess = self._session_bufs[session_id]

    # Same pipeline as _run_loop
    frame    = enhance_frame(frame)
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.web_holistic.process(rgb)

    hands_visible = (results.left_hand_landmarks  is not None or
                     results.right_hand_landmarks is not None)

    if not hands_visible:
        sess["no_hands"] += 1
        if sess["no_hands"] > 10:
            sess["frame_buf"].clear()
            sess["pred_buf"].clear()
            sess["conf_buf"].clear()
        return None, 0.0
    else:
        sess["no_hands"] = 0

    kps = extract_kps_from_results(results)
    sess["frame_buf"].append(kps)

    if len(sess["frame_buf"]) < 4:
        return None, 0.0

    seq = build_seq_from_buffer(sess["frame_buf"], self.seq_len)
    x   = torch.tensor(seq).unsqueeze(0)

    with torch.no_grad():
        probs = softmax(self.model(x).numpy().squeeze(0))

    top_idx  = int(np.argmax(probs))
    top_conf = float(probs[top_idx])

    if top_conf >= self.conf_thresh:
        sess["pred_buf"].append(top_idx)
        sess["conf_buf"].append(top_conf)
    else:
        sess["pred_buf"].append(None)
        sess["conf_buf"].append(0.0)

    pred_idx, vote_frac = smooth_prediction(sess["pred_buf"], sess["conf_buf"])

    if pred_idx is not None:
        return self.labels[pred_idx], vote_frac
    return None, top_conf