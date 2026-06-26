"""
capture.py — 画面捕获模块

支持三种模式：
  "dxcam" — DXGI 屏幕截帧（和 OBS 同一 API，推荐）
  "obs"   — OBS 虚拟摄像头
  "mss"   — GDI 截屏（兼容性最好）

自动回退：dxcam 失败 → mss → OBS 摄像头
"""

import time
import numpy as np
import cv2

from .config import (
    CAPTURE_MODE, OBS_CAMERA_INDEX, CAPTURE_MONITOR,
    CAPTURE_REGION, TARGET_FPS, CAPTURE_CROP,
)


class ScreenCapturer:
    def __init__(self):
        self.backend = None
        self.dxcam_cam = None
        self.mss_sct = None
        self.obs_cap = None
        self.region = CAPTURE_REGION
        self._started = False
        self._init()

    def _init(self):
        mode = CAPTURE_MODE

        if mode == "dxcam":
            if self._try_dxcam():
                return
            print("[Capture] dxcam 失败，回退 mss...")
            if self._try_mss():
                return
        elif mode == "mss":
            if self._try_mss():
                return
        elif mode == "obs":
            if self._try_obs():
                return

        # 全部失败，尝试 OBS 作为最后手段
        print("[Capture] 所有模式失败，尝试 OBS 虚拟摄像头...")
        if self._try_obs():
            return

        raise RuntimeError("无法初始化任何画面捕获方式！")

    def _try_dxcam(self):
        try:
            import dxcam
            self.dxcam_cam = dxcam.create(
                device_idx=CAPTURE_MONITOR, output_color="RGB",
            )
            self.backend = "dxcam"
            print("[Capture] ✅ dxcam (DXGI 屏幕截帧)")
            return True
        except Exception as e:
            print(f"[Capture] dxcam: {e}")
            return False

    def _try_mss(self):
        try:
            import mss
            self.mss_sct = mss.mss()
            self.backend = "mss"
            print("[Capture] ✅ mss (GDI 截屏)")
            return True
        except Exception as e:
            print(f"[Capture] mss: {e}")
            return False

    def _try_obs(self):
        for idx in range(4):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.obs_cap = cap
                    self.backend = "obs"
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[Capture] ✅ OBS 虚拟摄像头 (摄像头 {idx}, {w}x{h})")
                    return True
                cap.release()
        return False

    def start(self):
        if self._started:
            return
        if self.backend == "dxcam":
            self.dxcam_cam.start(
                region=self.region,
                target_fps=TARGET_FPS,
                video_mode=True,
            )
        self._started = True

    def stop(self):
        if not self._started:
            return
        if self.backend == "dxcam":
            self.dxcam_cam.stop()
        elif self.backend == "mss":
            self.mss_sct.close()
        elif self.backend == "obs":
            self.obs_cap.release()
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def grab_frame(self):
        if self.backend == "dxcam":
            frame = self.dxcam_cam.get_latest_frame()
            if frame is not None:
                return frame
        elif self.backend == "mss":
            return self._grab_mss()
        elif self.backend == "obs":
            ret, frame = self.obs_cap.read()
            if ret:
                return frame
        return None

    def _grab_mss(self):
        from PIL import Image
        monitors = self.mss_sct.monitors
        mon_idx = CAPTURE_MONITOR + 1
        monitor = monitors[mon_idx] if mon_idx < len(monitors) else monitors[1]
        if self.region:
            l, t, r, b = self.region
            monitor = {"left": l, "top": t, "width": r - l, "height": b - t}
        sct = self.mss_sct.grab(monitor)
        img = Image.frombytes("RGB", sct.size, sct.bgra, "raw", "BGRX")
        return np.array(img)
