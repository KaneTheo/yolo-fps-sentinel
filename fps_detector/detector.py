"""
detector.py — YOLO26 检测器

使用 PyTorch + CUDA（GPU 加速），自动回退到 CPU。
YOLO26 相比 YOLO11 去掉了 NMS 和 DFL，端到端更快更准。
"""

from dataclasses import dataclass, field
import numpy as np
import torch
from ultralytics import YOLO

from .config import (
    MODEL_NAME, CONFIDENCE_THRESHOLD, DETECT_CLASSES,
    INFERENCE_IMGSZ, HEAD_ZONE_RATIO, CAPTURE_CROP,
    BOTTOM_STRIP_RATIO, HANDS_CENTER_Y_RATIO, HANDS_BOX_HEIGHT_RATIO,
    HANDS_TOP_EDGE_RATIO,
)


@dataclass
class Detection:
    """单个检测结果，包含瞄准点信息"""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: int = -1
    _screen_cx: int = field(default=960, repr=False)
    _screen_cy: int = field(default=540, repr=False)
    _snap_dist: float = field(default=-1.0, repr=False)

    @property
    def center(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def head_box(self):
        """头部区域：框顶部的 20%"""
        head_h = max(int(self.height * HEAD_ZONE_RATIO), 10)
        cx = (self.x1 + self.x2) // 2
        half_w = max(self.width // 4, 10)
        return (cx - half_w, self.y1, cx + half_w, self.y1 + head_h)

    @property
    def snap_point(self):
        """瞄准点：头部区域中心"""
        hx1, hy1, hx2, hy2 = self.head_box
        return ((hx1 + hx2) // 2, (hy1 + hy2) // 2)

    @property
    def distance_to_center(self):
        """到屏幕中心的距离"""
        if self._snap_dist >= 0.0:
            return self._snap_dist
        sx, sy = self.snap_point
        return ((sx - self._screen_cx) ** 2 + (sy - self._screen_cy) ** 2) ** 0.5


class Detector:
    """YOLO26 PyTorch CUDA 检测器"""

    def __init__(self, screen_size=(1920, 1080)):
        self._screen_w, self._screen_h = screen_size

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Detector] 加载 YOLO26: {MODEL_NAME} | 设备: {device}")
        self._model = YOLO(MODEL_NAME)
        self._model.to(device)

        gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        crop_info = f"crop={CAPTURE_CROP}px " if CAPTURE_CROP else "全屏"
        print(
            f"[Detector] ✅ {gpu_name} | {crop_info}"
            f"imgsz={INFERENCE_IMGSZ} | 屏幕 {self._screen_w}x{self._screen_h}"
        )

    # ── 手部/武器过滤 ────────────────────────────────────────

    def _is_hand(self, x1, y1, x2, y2, frame_h):
        """过滤屏幕下方自己的手和武器"""
        cy_r = (y1 + y2) / 2 / frame_h
        bh_r = (y2 - y1) / frame_h
        y1_r = y1 / frame_h

        # 框在画面底部
        if cy_r > BOTTOM_STRIP_RATIO:
            return True
        # 框高度大且中心偏下
        if bh_r > HANDS_BOX_HEIGHT_RATIO and cy_r > HANDS_CENTER_Y_RATIO:
            return True
        # 框顶部在画面下半部
        if y1_r > HANDS_TOP_EDGE_RATIO:
            return True
        return False

    # ── 中心裁剪 ────────────────────────────────────────────

    def _crop_center(self, frame):
        """从画面中心裁出正方形"""
        if not CAPTURE_CROP or CAPTURE_CROP <= 0:
            return frame, 0, 0
        h, w = frame.shape[:2]
        crop = min(CAPTURE_CROP, h, w)
        x0 = (w - crop) // 2
        y0 = (h - crop) // 2
        return frame[y0:y0 + crop, x0:x0 + crop], x0, y0

    # ── 主接口 ──────────────────────────────────────────────

    def detect(self, frame):
        """输入 RGB 帧，返回 Detection 列表（按距离排序）"""
        frame_inp, off_x, off_y = self._crop_center(frame)

        # YOLO26 推理
        with torch.inference_mode():
            results = self._model(
                frame_inp,
                conf=CONFIDENCE_THRESHOLD,
                classes=DETECT_CLASSES,
                imgsz=INFERENCE_IMGSZ,
                verbose=False,
                half=True,  # FP16 加速
            )

        cx = self._screen_w // 2
        cy = self._screen_h // 2
        detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                bxy = list(map(int, box.xyxy[0].tolist()))
                conf = float(box.conf[0])

                x1 = bxy[0] + off_x
                y1 = bxy[1] + off_y
                x2 = bxy[2] + off_x
                y2 = bxy[3] + off_y

                if self._is_hand(x1, y1, x2, y2, self._screen_h):
                    continue

                detections.append(
                    Detection(x1, y1, x2, y2, conf, _screen_cx=cx, _screen_cy=cy)
                )

        # 按距离排序：离屏幕中心最近的排第一
        detections.sort(key=lambda d: d.distance_to_center)
        return detections
