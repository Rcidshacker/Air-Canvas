import cv2
import numpy as np
import mediapipe as mp
import torch
import os
from collections import deque
from threading import Thread, Lock, Event
from depth_anything_v2.dpt import DepthAnythingV2
from dataclasses import dataclass, field
from typing import List, Tuple
import logging
import yaml

@dataclass
class Config:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_dir: str = "./model_weights"
    model_filename: str = "depth_anything_v2_vits.pth"
    model_type: str = 'vits'
    input_size: int = 256
    pen_down_threshold: int = 120
    window_width: int = 1280
    window_height: int = 720
    max_points: int = 1024
    brush_sizes: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    colors: List[Tuple] = field(default_factory=lambda: [
        (255, 255, 255),  # White
        (0, 0, 0),        # Black
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0)     # Yellow
    ])

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        if os.path.exists(path):
            with open(path, 'r') as f:
                return cls(**yaml.safe_load(f))
        return cls()

class SharedState:
    def __init__(self):
        self.frame = None
        self.depth_display = None
        self.frame_lock = Lock()
        self.depth_lock = Lock()
        self.stop_event = Event()

class AirPainting:
    def __init__(self, config: Config):
        self.config = config
        self.state = SharedState()
        self.points = [deque(maxlen=config.max_points) for _ in range(len(config.colors))]
        self.color_index = 0
        self.brush_size_index = 0
        self.paint_window = self._init_canvas()
        
        self._setup_logging()
        self._init_model()
        self._init_mediapipe()
        self._start_depth_thread()

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')

    def _init_canvas(self):
        return np.ones((self.config.window_height, self.config.window_width, 3), 
                      dtype=np.uint8) * 255

    def _init_model(self):
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }

        model_path = os.path.join(self.config.model_dir, self.config.model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        try:
            self.depth_model = DepthAnythingV2(**model_configs[self.config.model_type])
            self.depth_model.load_state_dict(
                torch.load(model_path, map_location='cpu', weights_only=True)
            )
            self.depth_model = self.depth_model.to(self.config.device).eval()
        except Exception as e:
            logging.error(f"Model initialization failed: {e}")
            raise

    def _init_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def _depth_estimation_worker(self):
        while not self.state.stop_event.is_set():
            with self.state.frame_lock:
                current_frame = self.state.frame.copy() if self.state.frame is not None else None

            if current_frame is not None:
                try:
                    with torch.no_grad():
                        depth = self.depth_model.infer_image(current_frame, self.config.input_size)
                        depth = (depth - depth.min()) / (depth.max() - depth.min())
                        depth_colored = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                        
                    with self.state.depth_lock:
                        self.state.depth_display = depth_colored
                except Exception as e:
                    logging.error(f"Depth processing error: {e}")

    def _start_depth_thread(self):
        self.depth_thread = Thread(target=self._depth_estimation_worker, daemon=True)
        self.depth_thread.start()

    def _draw_ui(self):
        # Draw the color palette and brush sizes on the canvas.
        palette_start = self.config.window_width - 100
        for idx, color in enumerate(self.config.colors):
            y_start = 50 + idx * 60
            cv2.rectangle(self.paint_window, 
                         (palette_start, y_start),
                         (palette_start + 50, y_start + 50),
                         color, -1)
            if idx == self.color_index:
                cv2.rectangle(self.paint_window,
                            (palette_start - 2, y_start - 2),
                            (palette_start + 52, y_start + 52),
                            (0, 255, 0), 2)

        for idx, size in enumerate(self.config.brush_sizes):
            y_start = 50 + (len(self.config.colors) + idx + 1) * 60
            cv2.circle(self.paint_window,
                      (palette_start + 25, y_start + 25),
                      size // 2,
                      (0, 0, 0), -1)
            if idx == self.brush_size_index:
                cv2.circle(self.paint_window,
                          (palette_start + 25, y_start + 25),
                          size // 2 + 2,
                          (0, 255, 0), 2)

        cv2.rectangle(self.paint_window,
                     (palette_start, 10),
                     (palette_start + 50, 40),
                     (200, 200, 200), -1)
        cv2.putText(self.paint_window, "Clear",
                   (palette_start + 5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def _handle_interaction(self, hand_landmarks):
        index_tip = (int(hand_landmarks[8].x * self.config.window_width),
                     int(hand_landmarks[8].y * self.config.window_height))
        thumb_tip = (int(hand_landmarks[4].x * self.config.window_width),
                     int(hand_landmarks[4].y * self.config.window_height))

        pinch_dist = np.linalg.norm(np.array(index_tip) - np.array(thumb_tip))

        # UI interaction
        if index_tip[0] > self.config.window_width - 100:
            # Color selection
            for idx, _ in enumerate(self.config.colors):
                y_start = 50 + idx * 60
                if y_start <= index_tip[1] <= y_start + 50:
                    self.color_index = idx
                    return

            # Brush size selection
            for idx, _ in enumerate(self.config.brush_sizes):
                y_start = 50 + (len(self.config.colors) + idx + 1) * 60
                if y_start <= index_tip[1] <= y_start + 50:
                    self.brush_size_index = idx
                    return

            # Clear button
            if 10 <= index_tip[1] <= 40:
                self.paint_window = self._init_canvas()
                for q in self.points:
                    q.clear()
                return

        # Drawing: Use "pen down" when pinch distance is greater than the threshold.
        if pinch_dist > self.config.pen_down_threshold and index_tip[0] < self.config.window_width - 100:
            self.points[self.color_index].appendleft(index_tip)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.window_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.window_height)

        if not cap.isOpened():
            raise RuntimeError("Could not access webcam")

        try:
            while True:
                ret, raw_frame = cap.read()
                if not ret:
                    break

                with self.state.frame_lock:
                    self.state.frame = cv2.flip(raw_frame, 1)
                    display_frame = self.state.frame.copy()

                self._draw_ui()

                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            display_frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        self._handle_interaction(hand_landmarks.landmark)

                # Draw lines
                for i, (color, pts) in enumerate(zip(self.config.colors, self.points)):
                    for j in range(1, len(pts)):
                        if pts[j - 1] and pts[j]:
                            cv2.line(self.paint_window, pts[j - 1], pts[j],
                                     color, self.config.brush_sizes[self.brush_size_index])

                # Display outputs
                with self.state.depth_lock:
                    if self.state.depth_display is not None:
                        depth_display = cv2.resize(
                            self.state.depth_display,
                            (self.config.window_width, self.config.window_height)
                        )
                        cv2.imshow("Depth View", depth_display)

                cv2.imshow("Webcam View", display_frame)
                cv2.imshow("Painting Canvas", self.paint_window)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.state.stop_event.set()
            self.depth_thread.join()
            cap.release()
            cv2.destroyAllWindows()

def main():
    config = Config.from_yaml('config.yaml')
    app = AirPainting(config)
    app.run()

if __name__ == "__main__":
    main()
