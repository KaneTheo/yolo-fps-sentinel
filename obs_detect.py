"""
OBS 虚拟摄像头 + YOLO 人物检测
========================================
使用 OBS 的虚拟摄像头捕获桌面/游戏画面，
用 YOLO 检测画面中的人物（FPS 游戏角色），
并显示检测框。

使用方法：
  1. 打开 OBS → 启动虚拟摄像头（工具 → 虚拟摄像头）
  2. python obs_detect.py
  3. 按 Q 或 ESC 退出，按 F1 开关检测框

关于 OBS 虚拟摄像头设置：
  - OBS → 设置 → 虚拟摄像头 → 输出类型选"共享内存"或"DirectShow"
  - 场景中添加"游戏采集"或"显示器采集"
"""

import time as _time
from collections import deque
from ultralytics import YOLO
import cv2
import ctypes

# ============================================================
# 配置参数（根据你的需求调整）
# ============================================================

# OBS 虚拟摄像头通常在索引 1 或 2（0 是真实摄像头）
# 如果找不到，可以逐个尝试 0, 1, 2, 3...
OBS_CAMERA_INDEX = 1

# 检测置信度阈值（0-1），低于此值的检测结果忽略
# FPS 游戏建议 0.1-0.2（远距离目标低置信度也要检测到）
CONFIDENCE_THRESHOLD = 0.15

# 处理分辨率：为了速度，把画面缩放后再检测
# 640=快，960=平衡，1280=更准
PROCESS_WIDTH = 640

# 显示窗口的大小（缩小以避免遮挡游戏画面）
DISPLAY_SCALE = 0.5

# ============================================================
# YOLO 模型加载
# ============================================================
print("正在加载 YOLO26n 模型...")
model = YOLO("yolo26n.pt")
print("模型加载完成!\n")

# ============================================================
# OBS 虚拟摄像头连接
# ============================================================
print("正在连接 OBS 虚拟摄像头...")
print("（如果失败，请确保 OBS 虚拟摄像头已启动）")
print(f"（目标摄像头索引: {OBS_CAMERA_INDEX}，可在脚本顶部修改）\n")

cap = None
for idx in range(4):
    test_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if test_cap.isOpened():
        ret, _ = test_cap.read()
        if ret:
            print(f"  [摄像头 {idx}] ✅ 可用")
            cap = test_cap
            if idx == OBS_CAMERA_INDEX:
                break
            test_cap.release()
        else:
            print(f"  [摄像头 {idx}] ❌ 无法读取画面")
            test_cap.release()
    else:
        print(f"  [摄像头 {idx}] ❌ 打不开")

if cap is None:
    print("\n未找到 OBS 虚拟摄像头，尝试使用摄像头 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ 无法打开任何摄像头！")
    print("请确认：")
    print("  1. OBS 已启动虚拟摄像头（工具 → 虚拟摄像头 → 启动）")
    print("  2. 修改脚本顶部的 OBS_CAMERA_INDEX")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\n✅ 摄像头分辨率: {actual_w} x {actual_h}\n")

# ============================================================
# 创建检测结果窗口
# ============================================================
window_name = "YOLO 检测 - Q/ESC退出 | F1开关 | 可拖到副屏"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(
    window_name,
    int(actual_w * DISPLAY_SCALE),
    int(actual_h * DISPLAY_SCALE),
)

# ============================================================
# 状态变量
# ============================================================
show_detection = True
fps_queue = deque(maxlen=30)
last_f1_time = 0.0
last_esc_time = 0.0

# ============================================================
# 主循环
# ============================================================
print("开始检测！按 F1 开关检测框，按 Q/ESC 退出")
print("-" * 50)

while True:
    # 1. 从 OBS 虚拟摄像头读取一帧
    ret, frame = cap.read()
    if not ret:
        print("读取画面失败，重试...")
        _time.sleep(0.01)
        continue

    t_start = _time.perf_counter()

    # 2. 缩放画面以加速推理
    h, w = frame.shape[:2]
    scale = PROCESS_WIDTH / w
    frame_small = cv2.resize(frame, (PROCESS_WIDTH, int(h * scale)))

    # 3. YOLO 推理
    results = model(frame_small, verbose=False)
    result = results[0]

    # 4. 过滤 person 类，坐标映射回原始尺寸
    person_boxes = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id == 0:  # COCO: 0 = person
                coords = box.xyxy[0].tolist()
                x1 = int(coords[0] / scale)
                y1 = int(coords[1] / scale)
                x2 = int(coords[2] / scale)
                y2 = int(coords[3] / scale)
                conf = float(box.conf[0])
                if conf >= CONFIDENCE_THRESHOLD:
                    person_boxes.append((x1, y1, x2, y2, conf))

    # 5. 画检测框
    display_frame = frame.copy()
    if show_detection:
        for x1, y1, x2, y2, conf in person_boxes:
            # 颜色：高置信度绿，低置信度橙
            if conf > 0.5:
                color = (0, 255, 0)
            elif conf > 0.25:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # 置信度标签
            label = f"{conf:.0%}"
            cv2.putText(
                display_frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

            # 头部区域（顶部 20%）+ 瞄准点
            head_h = max(int((y2 - y1) * 0.2), 5)
            head_cx = (x1 + x2) // 2
            head_w = max((x2 - x1) // 4, 5)
            cv2.rectangle(
                display_frame,
                (head_cx - head_w, y1),
                (head_cx + head_w, y1 + head_h),
                (0, 0, 255), 1,
            )
            cv2.circle(
                display_frame,
                (head_cx, y1 + head_h // 2),
                4, (0, 0, 255), -1,
            )

    # 6. FPS 计算
    t_end = _time.perf_counter()
    fps = 1.0 / (t_end - t_start) if t_end > t_start else 0
    fps_queue.append(fps)
    avg_fps = sum(fps_queue) / len(fps_queue)

    # 7. 叠加信息
    cv2.putText(
        display_frame,
        f"FPS: {avg_fps:.1f} | 人数: {len(person_boxes)} | "
        f"{'[检测框 ON]' if show_detection else '[检测框 OFF]'}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )

    # 8. 显示
    cv2.imshow(
        window_name,
        cv2.resize(
            display_frame,
            (int(actual_w * DISPLAY_SCALE), int(actual_h * DISPLAY_SCALE)),
        ),
    )

    # 9. 键盘控制
    key = cv2.waitKey(1) & 0xFF
    now = _time.perf_counter()

    if key == ord("q"):
        print("按下 Q，退出...")
        break

    # F1 切换检测框（防抖 0.5 秒）
    if (ctypes.windll.user32.GetAsyncKeyState(0x70) & 0x8000) and (now - last_f1_time > 0.5):
        show_detection = not show_detection
        print(f"检测框: {'ON' if show_detection else 'OFF'}")
        last_f1_time = now

    # ESC 退出（防抖 0.5 秒）
    if (ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000) and (now - last_esc_time > 0.5):
        print("按下 ESC，退出...")
        break

# ============================================================
# 清理
# ============================================================
cap.release()
cv2.destroyAllWindows()
print("程序已退出。")
