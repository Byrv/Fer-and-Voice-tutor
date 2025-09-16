import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
from collections import deque
from appstate import AppState

class StudyModeFER:
    def __init__(self, state: AppState, model_path="models/emotion-ferplus-12-int8.onnx"):
        self.state = state
        self.LABELS = [
            "neutral", "happiness", "surprise", "sadness",
            "anger", "disgust", "fear", "contempt"
        ]

        self.DETECTION_CONF_FACE = 0.60
        self.PICK_LARGEST_FACE = True

        self.BOX_COLOR = (0, 255, 0)
        self.TEXT_COLOR = (0, 255, 0)

        self.current_emotion = None

        # ONNX session
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape
        self.input_type = inp.type

        _, c, h, w = self.input_shape
        if not (c == 1 and h == 64 and w == 64):
            raise RuntimeError(f"Model expects Nx1x64x64; got {self.input_shape}")

        mp_face = mp.solutions.face_detection
        self.face_detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=self.DETECTION_CONF_FACE
        )

        _ = self.session.run(None, {self.input_name: np.zeros((1, 1, 64, 64), np.float32)})

        # --- Confusion detection buffers ---
        self.window_size = 15  # ~0.5s if 30 FPS
        self.emotion_window = deque(maxlen=self.window_size)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        ex = np.exp(x, dtype=np.float64)
        return (ex / (np.sum(ex) + 1e-9)).astype(np.float32)

    @staticmethod
    def clamp_bbox(x1, y1, x2, y2, w, h):
        x1c = max(0, min(x1, w - 1))
        y1c = max(0, min(y1, h - 1))
        x2c = max(0, min(x2, w))
        y2c = max(0, min(y2, h))
        return x1c, y1c, x2c, y2c

    def pick_largest_detection(self, detections, frame_w, frame_h):
        best, best_area = None, -1
        for det in detections:
            if det.score and det.score[0] < self.DETECTION_CONF_FACE:
                continue
            bbox = det.location_data.relative_bounding_box
            x1 = int(bbox.xmin * frame_w)
            y1 = int(bbox.ymin * frame_h)
            x2 = x1 + int(bbox.width * frame_w)
            y2 = y1 + int(bbox.height * frame_h)
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > best_area:
                best_area, best = area, det
        return best

    def preprocess_face(self, face_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        arr = gray.astype(np.float32) if "uint8" not in str(self.input_type).lower() else gray.astype(np.uint8)
        return np.expand_dims(arr, axis=(0, 1))

    def detect_confusion(self, probs: np.ndarray) -> float:
        """Return a confusion score (0-1) based on weighted emotions."""
        # weights tuned from DAiSEE research papers
        surprise = probs[self.LABELS.index("surprise")]
        fear = probs[self.LABELS.index("fear")]
        sadness = probs[self.LABELS.index("sadness")]

        confusion_score = 0.4*surprise + 0.3*fear + 0.3*sadness
        return float(confusion_score)

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera read failed.")
                    break

                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detector.process(rgb)

                if results and results.detections:
                    detections = results.detections
                    if self.PICK_LARGEST_FACE and len(detections) > 1:
                        det = self.pick_largest_detection(detections, w, h)
                        detections = [det] if det is not None else []

                    for det in detections:
                        if det is None: continue
                        bbox = det.location_data.relative_bounding_box
                        x1 = int(bbox.xmin * w)
                        y1 = int(bbox.ymin * h)
                        x2 = x1 + int(bbox.width * w)
                        y2 = y1 + int(bbox.height * h)
                        x1, y1, x2, y2 = self.clamp_bbox(x1, y1, x2, y2, w, h)
                        if x2 <= x1 or y2 <= y1: continue

                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size == 0: continue

                        x = self.preprocess_face(face_crop)
                        logits = self.session.run(None, {self.input_name: x})[0][0]
                        probs = self.softmax(logits)

                        # Append to moving window
                        self.emotion_window.append(probs)

                        # Average probs over window
                        avg_probs = np.mean(self.emotion_window, axis=0)

                        # Get top emotion
                        top_idx = int(np.argmax(avg_probs))
                        top_prob = float(avg_probs[top_idx])
                        label = self.LABELS[top_idx]

                        # Compute confusion score
                        confusion_score = self.detect_confusion(avg_probs)

                        # --- Trigger logic ---
                        if confusion_score > 0.15:  # tuned threshold
                            if self.current_emotion != "confused":
                                self.current_emotion = "confused"
                                self.state.trigger_interrupt("confused")
                        elif label == "happiness" and top_prob > 0.6:
                            if self.current_emotion != "happy":
                                self.current_emotion = "happy"
                                self.state.trigger_interrupt("happy")
                        else:
                            self.current_emotion = label

                        # Draw
                        cv2.rectangle(frame, (x1, y1), (x2, y2), self.BOX_COLOR, 2)
                        cv2.putText(frame,
                                    f"{label} ({top_prob:.2f}) | Conf: {confusion_score:.2f}",
                                    (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 2)

                cv2.imshow("StudyModeFER", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
