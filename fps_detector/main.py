"""
main.py — FPS 人物检测主程序
============================================================

架构（三线程流水线）：
  CaptureThread   — 截屏帧                      → frame_queue
  InferenceThread — YOLO26 推理                  → detection_queue
  Main Thread     — Tracker + Overlay 透明浮窗

快捷键：
  F1  — 开/关检测框
  F2  — 退出程序
  F10 — 保存截图

用法：
  python -m fps_detector.main
"""

import time
import threading
import queue
import ctypes
import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fps_detector.capture import ScreenCapturer
from fps_detector.detector import Detector
from fps_detector.overlay import Overlay
from fps_detector.tracker import ByteTracker
from fps_detector.aim import AimAssist
from fps_detector.config import (
    TOGGLE_KEY, EXIT_KEY, SNAPSHOT_KEY,
    TARGET_FPS, INFERENCE_QUEUE_SIZE, DETECTION_QUEUE_SIZE,
    AIM_TRIGGER_KEY, AIM_SMOOTH, AIM_SPEED, AIM_FOV, AIM_ENABLED,
    AIM_DEADZONE, AIM_AIM_RATIO,
)

# ── 按键检测（Windows GetAsyncKeyState）─────────────────────

_VK = {
    "f9": 0x78, "f10": 0x79, "esc": 0x1B,
    "f1": 0x70, "f2": 0x71, "f3": 0x72,
}


def _key_pressed(name):
    vk = _VK.get(name.lower())
    if vk is None:
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


_key_prev = {}


def _key_just_pressed(name):
    cur = _key_pressed(name)
    prev = _key_prev.get(name, False)
    _key_prev[name] = cur
    return cur and not prev


# ── 全局状态 ────────────────────────────────────────────────

_enabled = True
_running = True
_request_snapshot = False

_fps_lock = threading.Lock()
_fps_counts = {"cap": 0, "inf": 0, "ovl": 0}

_frame_queue = queue.Queue(maxsize=INFERENCE_QUEUE_SIZE)
_detection_queue = queue.Queue(maxsize=DETECTION_QUEUE_SIZE)


# ── 线程函数 ────────────────────────────────────────────────

def _capture_loop(capturer):
    interval = 1.0 / TARGET_FPS
    while _running:
        t0 = time.perf_counter()
        frame = capturer.grab_frame()
        if frame is not None:
            if _frame_queue.full():
                try:
                    _frame_queue.get_nowait()
                except queue.Empty:
                    pass
            _frame_queue.put_nowait(frame)
            with _fps_lock:
                _fps_counts["cap"] += 1
        sleep = interval - (time.perf_counter() - t0)
        if sleep > 0:
            time.sleep(sleep)


def _inference_loop(detector):
    while _running:
        try:
            frame = _frame_queue.get(timeout=0.1)
            # 跳过积压旧帧，始终推理最新帧
            while True:
                try:
                    frame = _frame_queue.get_nowait()
                except queue.Empty:
                    break
        except queue.Empty:
            continue

        detections = detector.detect(frame)
        if _detection_queue.full():
            try:
                _detection_queue.get_nowait()
            except queue.Empty:
                pass
        _detection_queue.put_nowait((frame, detections))
        with _fps_lock:
            _fps_counts["inf"] += 1


# ── 主入口 ──────────────────────────────────────────────────

def main():
    global _enabled, _running, _request_snapshot

    print("=" * 55)
    print("  FPS Detector — YOLO26 + ByteTrack + 透明浮窗")
    print("=" * 55)

    capturer = ScreenCapturer()
    print(f"[Main] 截帧模式: {capturer.backend}")

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    print(f"[Main] 屏幕分辨率: {sw}x{sh}")

    detector = Detector(screen_size=(sw, sh))
    overlay = Overlay()

    print(f"[Main] 快捷键: {TOGGLE_KEY.upper()}=开关 | "
          f"{SNAPSHOT_KEY.upper()}=截图 | {EXIT_KEY.upper()}=退出")

    last_frame = None
    last_detections = []
    last_raw_count = 0
    cap_size = None
    fps = {"cap": 0.0, "inf": 0.0, "ovl": 0.0}
    ovl_count = 0
    fps_timer = time.perf_counter()
    tracker = ByteTracker()
    aim = AimAssist(
        trigger_key=AIM_TRIGGER_KEY,
        smooth=AIM_SMOOTH,
        speed=AIM_SPEED,
        fov=AIM_FOV,
        deadzone=AIM_DEADZONE,
        aim_ratio=AIM_AIM_RATIO,
    ) if AIM_ENABLED else None

    with capturer:
        for thr in [
            threading.Thread(target=_capture_loop, args=(capturer,),
                             daemon=True, name="Capture"),
            threading.Thread(target=_inference_loop, args=(detector,),
                             daemon=True, name="Inference"),
        ]:
            thr.start()

        print("[Main] 运行中... 按 F2 退出\n")

        while _running:
            t0 = time.perf_counter()

            # 按键
            if _key_just_pressed(TOGGLE_KEY):
                _enabled = not _enabled
                print(f"[Main] 浮窗: {'ON' if _enabled else 'OFF'}")

            if _key_just_pressed(EXIT_KEY):
                _running = False
                print("[Main] 退出...")
                break

            if _key_just_pressed(SNAPSHOT_KEY):
                _request_snapshot = True

            # 获取最新检测结果
            try:
                last_frame, raw_detections = _detection_queue.get_nowait()
                last_raw_count = len(raw_detections)
                if cap_size is None and last_frame is not None:
                    cap_size = (last_frame.shape[1], last_frame.shape[0])
                    if cap_size != (sw, sh):
                        print(f"[Main] 截帧 {cap_size[0]}x{cap_size[1]} "
                              f"→ 缩放至浮窗 {sw}x{sh}")
                last_detections = tracker.update(raw_detections, sw, sh)

                # 更新 aim 的坐标映射
                if aim is not None and cap_size is not None:
                    aim._cap_w, aim._cap_h = cap_size
            except queue.Empty:
                pass

            # 鼠标吸附
            if aim is not None and _enabled and last_detections:
                aim.update(last_detections)

            overlay.update(
                last_detections, fps["cap"], fps["inf"], fps["ovl"],
                _enabled, cap_size,
            )
            ovl_count += 1

            # 截图
            if _request_snapshot and last_frame is not None:
                _request_snapshot = False
                _save_snapshot(last_frame, last_detections)

            # 每秒打印 FPS 信息
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0:
                with _fps_lock:
                    counts = dict(_fps_counts)
                    _fps_counts.update({"cap": 0, "inf": 0, "ovl": 0})
                fps = {k: counts[k] / elapsed for k in counts}
                fps["ovl"] = ovl_count / elapsed
                ovl_count = 0
                fps_timer = time.perf_counter()

                det_info = "  ".join(
                    f"T{d.track_id}:{int(d.confidence * 100)}%/{int(d.distance_to_center)}px"
                    for d in last_detections
                ) if last_detections else "none"

                print(
                    f"[Main] Cap:{fps['cap']:.0f} Inf:{fps['inf']:.0f} "
                    f"Ovl:{fps['ovl']:.0f} | raw:{last_raw_count} "
                    f"trk:{len(last_detections)} | {det_info}"
                )

            spent = time.perf_counter() - t0
            sleep = (1.0 / TARGET_FPS) - spent
            if sleep > 0:
                time.sleep(sleep)

    overlay.destroy()
    print("[Main] 已退出")


def _save_snapshot(frame, detections):
    if frame is None:
        return
    try:
        from PIL import Image, ImageDraw
        import datetime
        from fps_detector.config import BOX_COLOR, PRIMARY_TARGET_COLOR, HEAD_ZONE_COLOR

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        for i, det in enumerate(detections):
            color = (
                tuple(int(PRIMARY_TARGET_COLOR[j:j + 2], 16) for j in (1, 3, 5))
                if i == 0
                else tuple(int(BOX_COLOR[j:j + 2], 16) for j in (1, 3, 5))
            )
            head_c = tuple(int(HEAD_ZONE_COLOR[j:j + 2], 16) for j in (1, 3, 5))
            draw.rectangle([det.x1, det.y1, det.x2, det.y2], outline=color, width=2)
            draw.rectangle(list(det.head_box), outline=head_c, width=1)
            sx, sy = det.snap_point
            draw.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=(255, 0, 0))
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"snapshot_{ts}.png"
        img.save(fname)
        print(f"[Snapshot] 已保存: {fname}")
    except Exception as e:
        print(f"[Snapshot] 保存失败: {e}")


if __name__ == "__main__":
    main()
