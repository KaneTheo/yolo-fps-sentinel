"""
aim.py — 鼠标吸附（Snap）

按住触发键 → 鼠标自动移向最近目标的胸部（不吸头，更自然）

关键机制（防转圈）：
  1. Deadzone：距离 < DEADZONE 像素时完全不动，消除震荡
  2. 距离衰减：越近移得越慢，不会冲过头
  3. 最小步长：微动不响应，消除微抖
"""

import math
import ctypes
from ctypes import wintypes

from .config import (
    AIM_TRIGGER_KEY, AIM_SMOOTH, AIM_SPEED, AIM_FOV,
    AIM_DEADZONE, AIM_AIM_RATIO,
)

# ── Windows SendInput ──────────────────────────────────────

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
wintypes.ULONG_PTR = wintypes.WPARAM


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", wintypes.ULONG_PTR),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", MOUSEINPUT),
    ]


def _send_mouse_move(dx, dy):
    inp = INPUT(
        type=INPUT_MOUSE,
        mi=MOUSEINPUT(dx=int(dx), dy=int(dy), mouseData=0,
                      dwFlags=MOUSEEVENTF_MOVE, time=0, dwExtraInfo=0),
    )
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def _get_cursor_pos():
    pt = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def _get_screen_size():
    return (
        ctypes.windll.user32.GetSystemMetrics(0),
        ctypes.windll.user32.GetSystemMetrics(1),
    )


def _key_pressed(vk_code):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)


VK_MAP = {
    "rmb": 0x02, "lmb": 0x01, "mmb": 0x04,
    "shift": 0x10, "ctrl": 0x11, "alt": 0x12,
    "x1": 0x05, "x2": 0x06, "caps": 0x14,
}


class AimAssist:
    def __init__(self, trigger_key="rmb", smooth=True, speed=0.6, fov=300,
                 deadzone=30, aim_ratio=0.45, cap_size=None):
        self.trigger_vk = VK_MAP.get(trigger_key.lower(), 0x02)
        self.smooth = smooth
        self.speed = speed
        self.fov = fov
        self.deadzone = deadzone    # 在这个像素范围内不移动
        self.aim_ratio = aim_ratio  # 瞄准点在 bbox 的什么位置（0.45=胸部）

        self._screen_w, self._screen_h = _get_screen_size()
        self._cap_w = cap_size[0] if cap_size else self._screen_w
        self._cap_h = cap_size[1] if cap_size else self._screen_h

        # 目标坐标 EMA：消除 YOLO bbox 抖动
        self._ema_x = None
        self._ema_y = None
        self._ema_tid = -1

        # 目标锁定：按住右键时锁定一个人，不松就不换
        self._locked_tid = -1      # 当前锁定的 track_id
        self._trigger_was_down = False  # 上一帧右键状态

        print(
            f"[Aim] 键: {trigger_key.upper()} | "
            f"{'平滑' if smooth else '瞬移'} | "
            f"速度: {speed} | FOV: {fov}px | 死区: {deadzone}px | "
            f"瞄准位: bbox {aim_ratio:.0%} | 锁定模式"
        )

    def _reset_ema(self):
        self._ema_x = None
        self._ema_y = None
        self._ema_tid = -1

    def _to_screen(self, x, y):
        rx = self._screen_w / self._cap_w
        ry = self._screen_h / self._cap_h
        return int(x * rx), int(y * ry)

    def _get_aim_point(self, det):
        """计算瞄准点：bbox 上部 aim_ratio 位置（默认 45% = 胸部）"""
        h = det.y2 - det.y1
        aim_y = det.y1 + int(h * self.aim_ratio)
        aim_x = (det.x1 + det.x2) // 2
        return aim_x, aim_y

    def update(self, detections):
        """带目标锁定的吸附更新"""
        trigger_down = _key_pressed(self.trigger_vk)
        trigger_just_pressed = trigger_down and not self._trigger_was_down
        self._trigger_was_down = trigger_down

        if not trigger_down:
            self._reset_ema()
            self._locked_tid = -1
            return False

        if not detections:
            self._reset_ema()
            return False

        # ── 目标选择：锁定优先 ──────────────────────────
        # 按下右键的那一刻 → 锁定最近的目标
        # 按住过程中 → 跟着锁定的目标，不换人
        # 锁定的目标消失 → 自动切换到新的最近目标

        target = None

        if trigger_just_pressed:
            # 刚按下：锁定最近且在 FOV 内的目标
            for d in detections:
                if self.fov <= 0 or d.distance_to_center <= self.fov:
                    target = d
                    self._locked_tid = d.track_id
                    break
            if target is None:
                self._locked_tid = -1
                return False
        else:
            # 按住中：找锁定的目标还在不在
            if self._locked_tid > 0:
                for d in detections:
                    if d.track_id == self._locked_tid:
                        target = d
                        break
            # 锁定的目标消失了 → 自动锁新的最近
            if target is None:
                for d in detections:
                    if self.fov <= 0 or d.distance_to_center <= self.fov:
                        target = d
                        self._locked_tid = d.track_id
                        break
            if target is None:
                self._reset_ema()
                return False

        # 原始瞄准点（捕获坐标 → 屏幕坐标）
        raw_ax, raw_ay = self._get_aim_point(target)
        raw_tx, raw_ty = self._to_screen(raw_ax, raw_ay)

        # ── EMA 平滑目标坐标 ──────────────────────────────
        if target.track_id != self._ema_tid or self._ema_x is None:
            self._ema_x = float(raw_tx)
            self._ema_y = float(raw_ty)
            self._ema_tid = target.track_id
        else:
            # 跳变检测：raw 与 EMA 差距超过阈值 → 大概率是
            # 头部↔全身 bbox 类型切换 → 用超低 alpha 压住跳变
            jump = math.hypot(raw_tx - self._ema_x, raw_ty - self._ema_y)
            if jump > 50:
                a = 0.08  # 大跳 → 几乎不动，等 YOLO 稳定
            elif jump > 20:
                a = 0.15  # 中跳
            else:
                a = 0.25  # 正常微抖
            self._ema_x = a * raw_tx + (1 - a) * self._ema_x
            self._ema_y = a * raw_ty + (1 - a) * self._ema_y

        target_x, target_y = self._ema_x, self._ema_y

        # 当前鼠标位置
        cur_x, cur_y = _get_cursor_pos()

        # 计算到 EMA 目标的距离
        dx = target_x - cur_x
        dy = target_y - cur_y
        dist = math.hypot(dx, dy)

        # Deadzone
        if dist <= self.deadzone:
            return False

        if self.smooth:
            # ── 平滑模式（CS 优化版）─────────────────────────
            # 远距离大跨步，近距离精细控制

            # 基础：每帧移剩余距离 × speed
            step = dist * self.speed

            # 自适应封顶：越远越快
            if dist > 150:
                MAX_STEP = 40   # 远距离大步追
            elif dist > 80:
                MAX_STEP = 30   # 中距离
            elif dist > 40:
                MAX_STEP = 18   # 近距精细
            else:
                MAX_STEP = 10   # 很近了，慢

            MIN_STEP = 2

            if step > MAX_STEP:
                step = MAX_STEP
            elif step < MIN_STEP and dist > self.deadzone:
                step = MIN_STEP

            mx = dx / dist * step
            my = dy / dist * step
        else:
            # ── 瞬移模式：直接跳 ──────────────────────────────
            mx = dx
            my = dy

        _send_mouse_move(mx, my)
        return True
