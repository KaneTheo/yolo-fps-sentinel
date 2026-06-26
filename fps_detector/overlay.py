"""
overlay.py — 透明浮窗（tkinter）

在游戏画面上叠加检测框，鼠标可穿透，不影响操作。
关键设计：
  1. 不闪烁 — 用 canvas.coords() 移动元素，不删除重建
  2. 鼠标穿透 — WS_EX_TRANSPARENT 让点击穿透到游戏
  3. 不会被截屏 — WDA_EXCLUDEFROMCAPTURE 让 dxcam 抓不到浮窗
"""

import tkinter as tk
import ctypes

from .config import (
    BOX_COLOR, PRIMARY_TARGET_COLOR, HEAD_ZONE_COLOR,
    SNAP_POINT_COLOR, SNAP_ZONE_COLOR, FPS_COLOR,
    BOX_WIDTH, LABEL_FONT_SIZE, FPS_FONT_SIZE,
    SNAP_ZONE_RADIUS, SHOW_SNAP_ZONE, BOX_SNAP_PX,
    OVERLAY_MAX_POOL_SIZE,
    PRIMARY_SWITCH_MARGIN, PRIMARY_HOLD_FRAMES, PRIMARY_ADVANTAGE_PX,
)

_TRANSPARENT_COLOR = "#010101"


class Overlay:
    def __init__(self):
        self._root = tk.Tk()
        self._setup_window()
        self._canvas = tk.Canvas(
            self._root, bg=_TRANSPARENT_COLOR, highlightthickness=0,
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._make_click_through()
        self._exclude_from_capture()
        self._root.update_idletasks()

        self._sw = self._root.winfo_screenwidth()
        self._sh = self._root.winfo_screenheight()

        self._track_items = {}
        self._free_pool = []
        self._primary_tid = -1
        self._primary_last_dist = float("inf")
        self._primary_lost_frames = 0
        self._last_box = {}
        self._last_label = {}
        self._init_static_items()

    def _init_static_items(self):
        cx, cy = self._sw // 2, self._sh // 2
        r = SNAP_ZONE_RADIUS or 200
        self._snap_circle = self._canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline=SNAP_ZONE_COLOR, width=1, dash=(4, 6), state="hidden",
        )
        self._ch_h = self._canvas.create_line(
            cx - 14, cy, cx + 14, cy, fill=SNAP_ZONE_COLOR, width=1, state="hidden",
        )
        self._ch_v = self._canvas.create_line(
            cx, cy - 14, cx, cy + 14, fill=SNAP_ZONE_COLOR, width=1, state="hidden",
        )
        self._fps_item = self._canvas.create_text(
            self._sw - 10, 10, text="", fill=FPS_COLOR,
            font=("Consolas", FPS_FONT_SIZE, "bold"), anchor="ne", state="hidden",
        )

    def _setup_window(self):
        r = self._root
        r.overrideredirect(True)
        r.attributes("-topmost", True)
        r.geometry(f"{r.winfo_screenwidth()}x{r.winfo_screenheight()}+0+0")
        r.wm_attributes("-transparentcolor", _TRANSPARENT_COLOR)
        r.configure(bg=_TRANSPARENT_COLOR)

    def _make_click_through(self):
        try:
            self._root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self._root.winfo_id())
            if hwnd == 0:
                hwnd = self._root.winfo_id()
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_LAYERED = 0x00080000
            GWL_EXSTYLE = -20
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE, style | WS_EX_TRANSPARENT | WS_EX_LAYERED,
            )
        except Exception as e:
            print(f"[Overlay] 鼠标穿透设置失败: {e}")

    def _exclude_from_capture(self):
        try:
            self._root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self._root.winfo_id())
            if hwnd == 0:
                hwnd = self._root.winfo_id()
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            result = ctypes.windll.user32.SetWindowDisplayAffinity(
                hwnd, WDA_EXCLUDEFROMCAPTURE,
            )
            if result:
                print("[Overlay] ✅ 已从截屏中排除")
            else:
                print("[Overlay] ⚠️ 排除截屏失败，浮窗可能被 dxcam 捕获")
        except Exception as e:
            print(f"[Overlay] 排除截屏失败: {e}")

    def _new_item_group(self):
        return {
            "box": self._canvas.create_rectangle(
                0, 0, 1, 1, outline=BOX_COLOR, width=BOX_WIDTH, state="hidden",
            ),
            "head": self._canvas.create_rectangle(
                0, 0, 1, 1, outline=HEAD_ZONE_COLOR, width=1,
                dash=(3, 3), state="hidden",
            ),
            "dot": self._canvas.create_oval(
                0, 0, 1, 1, fill=SNAP_POINT_COLOR,
                outline=SNAP_POINT_COLOR, state="hidden",
            ),
            "label": self._canvas.create_text(
                0, 0, text="", fill=BOX_COLOR,
                font=("Consolas", LABEL_FONT_SIZE, "bold"),
                anchor="sw", state="hidden",
            ),
        }

    def _acquire_item_group(self):
        if self._free_pool:
            return self._free_pool.pop()
        return self._new_item_group()

    def _release_item_group(self, items):
        cv = self._canvas
        if len(self._free_pool) >= OVERLAY_MAX_POOL_SIZE:
            for item_id in items.values():
                cv.delete(item_id)
        else:
            for item_id in items.values():
                cv.itemconfig(item_id, state="hidden")
            self._free_pool.append(items)

    def update(self, detections, fps_cap, fps_inf, fps_ovl,
               enabled=True, cap_size=None):
        cv = self._canvas

        if not enabled:
            cv.itemconfig(self._snap_circle, state="hidden")
            cv.itemconfig(self._ch_h, state="hidden")
            cv.itemconfig(self._ch_v, state="hidden")
            cv.itemconfig(self._fps_item, state="hidden")
            for tid in list(self._track_items):
                self._release_item_group(self._track_items.pop(tid))
                self._last_box.pop(tid, None)
                self._last_label.pop(tid, None)
            self._primary_tid = -1
            self._root.update()
            return

        cap_w = cap_size[0] if cap_size else self._sw
        cap_h = cap_size[1] if cap_size else self._sh
        rx = self._sw / cap_w
        ry = self._sh / cap_h

        def sx(x):
            return int(x * rx)

        def sy(y):
            return int(y * ry)

        cx_s, cy_s = self._sw // 2, self._sh // 2

        # 吸附圈 + 十字准星
        if SHOW_SNAP_ZONE and SNAP_ZONE_RADIUS > 0:
            r = SNAP_ZONE_RADIUS
            cv.coords(self._snap_circle, cx_s - r, cy_s - r, cx_s + r, cy_s + r)
            cv.itemconfig(self._snap_circle, state="normal", outline=SNAP_ZONE_COLOR)
            cv.coords(self._ch_h, cx_s - 14, cy_s, cx_s + 14, cy_s)
            cv.itemconfig(self._ch_h, state="normal")
            cv.coords(self._ch_v, cx_s, cy_s - 14, cx_s, cy_s + 14)
            cv.itemconfig(self._ch_v, state="normal")
        else:
            cv.itemconfig(self._snap_circle, state="hidden")
            cv.itemconfig(self._ch_h, state="hidden")
            cv.itemconfig(self._ch_v, state="hidden")

        fps_text = (
            f"Cap:{fps_cap:.0f}  Inf:{fps_inf:.0f}  Ovl:{fps_ovl:.0f}"
            f"  |  {len(detections)} targets"
        )
        cv.itemconfig(self._fps_item, text=fps_text, state="normal")

        active_ids = {d.track_id for d in detections}
        for tid in [k for k in self._track_items if k not in active_ids]:
            self._release_item_group(self._track_items.pop(tid))
            self._last_box.pop(tid, None)
            self._last_label.pop(tid, None)

        # 主目标选择
        prev_primary = self._primary_tid
        nearest = detections[0] if detections else None
        cur = next((d for d in detections if d.track_id == self._primary_tid), None)

        if cur is not None:
            self._primary_lost_frames = 0
            self._primary_last_dist = cur.distance_to_center
            if (nearest is not None
                    and nearest.distance_to_center < cur.distance_to_center - PRIMARY_SWITCH_MARGIN):
                self._primary_tid = nearest.track_id
                self._primary_last_dist = nearest.distance_to_center
        elif nearest is not None:
            self._primary_lost_frames += 1
            new_dist = nearest.distance_to_center
            if (new_dist < self._primary_last_dist - PRIMARY_ADVANTAGE_PX
                    or self._primary_lost_frames >= PRIMARY_HOLD_FRAMES):
                self._primary_tid = nearest.track_id
                self._primary_last_dist = new_dist
                self._primary_lost_frames = 0
        else:
            self._primary_tid = -1
            self._primary_last_dist = float("inf")
            self._primary_lost_frames = 0

        for det in detections:
            is_primary = (det.track_id == self._primary_tid)
            color = PRIMARY_TARGET_COLOR if is_primary else BOX_COLOR
            bw = BOX_WIDTH + (1 if is_primary else 0)
            dot_r = 4 if is_primary else 2

            is_new = det.track_id not in self._track_items
            if is_new:
                self._track_items[det.track_id] = self._acquire_item_group()
            items = self._track_items[det.track_id]

            bx1, by1 = sx(det.x1), sy(det.y1)
            bx2, by2 = sx(det.x2), sy(det.y2)
            last = self._last_box.get(det.track_id)
            coords_moved = (
                last is None
                or abs(bx1 - last[0]) > BOX_SNAP_PX
                or abs(by1 - last[1]) > BOX_SNAP_PX
                or abs(bx2 - last[2]) > BOX_SNAP_PX
                or abs(by2 - last[3]) > BOX_SNAP_PX
            )
            if coords_moved:
                cv.coords(items["box"], bx1, by1, bx2, by2)
                hx1, hy1, hx2, hy2 = det.head_box
                cv.coords(items["head"], sx(hx1), sy(hy1), sx(hx2), sy(hy2))
                spx, spy = sx(det.snap_point[0]), sy(det.snap_point[1])
                cv.coords(items["dot"],
                          spx - dot_r, spy - dot_r, spx + dot_r, spy + dot_r)
                cv.coords(items["label"], bx1 + 4, by1 - 2)
                self._last_box[det.track_id] = (bx1, by1, bx2, by2)

            primary_changed = (
                (det.track_id == self._primary_tid)
                != (det.track_id == prev_primary)
            )
            if is_new or primary_changed:
                cv.itemconfig(items["box"], outline=color, width=bw, state="normal")
                cv.itemconfig(items["head"], outline=HEAD_ZONE_COLOR, state="normal")
                cv.itemconfig(items["dot"], fill=SNAP_POINT_COLOR,
                              outline=SNAP_POINT_COLOR, state="normal")

            conf_r = int(round(det.confidence / 0.10) * 10)
            dist_r = int(round(det.distance_to_center / 30) * 30)
            label = f"{conf_r}%  {dist_r}px" if is_primary else f"{dist_r}px"

            label_changed = label != self._last_label.get(det.track_id)
            if is_new or label_changed or primary_changed:
                cv.itemconfig(items["label"], text=label, fill=color, state="normal")
                self._last_label[det.track_id] = label

        self._root.update()

    def destroy(self):
        try:
            self._root.destroy()
        except Exception:
            pass
