"""
tracker.py — ByteTrack + Kalman 多目标追踪

作用：
  1. 让检测框不抖动（Kalman 滤波平滑）
  2. 同一个人保持同一个 ID（不闪烁不消失）
  3. 过滤单帧误检（需要连续出现才显示）
  4. 短暂遮挡不丢目标（预测运动轨迹）

生命周期：
  YOLO检测 → TENTATIVE(暂存) → 连续N帧检测到 → CONFIRMED(显示)
  CONFIRMED → 连续N帧没检测到 → 删除
"""

import numpy as np
from .detector import Detection
from .config import (
    TRACKER_IOU_THRESH, TRACKER_HIGH_CONF, TRACKER_MAX_AGE,
    TRACKER_MIN_HITS, TRACKER_CENTER_DIST_FALLBACK, TRACKER_DEDUP_DIST,
    TRACKER_REID_DIST, TRACKER_REID_TTL, TRACKER_GHOST_MISS_LIMIT,
    HEAD_ZONE_RATIO, CONFIDENCE_THRESHOLD, NEW_TRACK_CONF,
)

_TENTATIVE = 0
_CONFIRMED = 1


# ── Kalman 滤波器 ────────────────────────────────────────────

class KalmanBoxFilter:
    """
    恒定速度模型 Kalman 滤波器。
    状态 [cx, cy, w, h, vx, vy] → 预测下一帧位置。
    """

    def __init__(self, cx, cy, w, h):
        self.x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float64)

        self.F = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)

        self.H = np.eye(4, 6, dtype=np.float64)
        self.P = np.diag([10., 10., 20., 20., 5000., 5000.]).astype(np.float64)
        self.Q = np.diag([1., 1., 2., 2., 800., 800.]).astype(np.float64)
        self.R = np.diag([5., 5., 10., 10.]).astype(np.float64)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        return self._to_xyxy()

    def update(self, cx, cy, w, h):
        z = np.array([cx, cy, w, h], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        return self._to_xyxy()

    def _to_xyxy(self):
        cx, cy, w, h = self.x[:4]
        return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)


# ── Track 对象 ───────────────────────────────────────────────

class _Track:
    _id_counter = 1

    def __init__(self, det, reuse_id=None):
        cx = (det.x1 + det.x2) * 0.5
        cy = (det.y1 + det.y2) * 0.5
        w = float(det.x2 - det.x1)
        h = float(det.y2 - det.y1)

        self.kf = KalmanBoxFilter(cx, cy, w, h)
        self.id = reuse_id if reuse_id is not None else _Track._id_counter
        if reuse_id is None:
            _Track._id_counter += 1

        self.state = _TENTATIVE
        self.conf = det.confidence
        self.hits = 1
        self.miss_streak = 0
        self._box = (float(det.x1), float(det.y1),
                     float(det.x2), float(det.y2))

        head_h0 = max(h * HEAD_ZONE_RATIO, 10.0)
        self._snap_ema = (cx, float(det.y1) + head_h0 * 0.5)

        if self.hits >= TRACKER_MIN_HITS:
            self.state = _CONFIRMED

    def predict(self):
        self._box = self.kf.predict()
        self.miss_streak += 1
        if self.miss_streak >= 1:
            self.kf.x[4] = 0.0
            self.kf.x[5] = 0.0

    def update(self, det):
        cx = (det.x1 + det.x2) * 0.5
        cy = (det.y1 + det.y2) * 0.5
        w = float(det.x2 - det.x1)
        h = float(det.y2 - det.y1)
        # 用 YOLO 原始坐标，不走 Kalman 滤波（CS 人物急停急动，恒速模型天然滞后）
        self._box = (float(det.x1), float(det.y1), float(det.x2), float(det.y2))
        # Kalman 照跑（遮挡预测用），但不影响显示坐标
        self.kf.update(cx, cy, w, h)
        self.conf = 0.12 * det.confidence + 0.88 * self.conf
        self.hits += 1
        self.miss_streak = 0
        if self.state == _TENTATIVE and self.hits >= TRACKER_MIN_HITS:
            self.state = _CONFIRMED

        dh = max(h * HEAD_ZONE_RATIO, 10.0)
        self._snap_ema = (
            0.85 * cx + 0.15 * self._snap_ema[0],
            0.85 * (float(det.y1) + dh * 0.5) + 0.15 * self._snap_ema[1],
        )

    def update_weak(self, det):
        cx = (det.x1 + det.x2) * 0.5
        cy = (det.y1 + det.y2) * 0.5
        w = float(det.x2 - det.x1)
        h = float(det.y2 - det.y1)
        self._box = (float(det.x1), float(det.y1), float(det.x2), float(det.y2))
        self.kf.update(cx, cy, w, h)
        dh = max(h * HEAD_ZONE_RATIO, 10.0)
        self._snap_ema = (
            0.85 * cx + 0.15 * self._snap_ema[0],
            0.85 * (float(det.y1) + dh * 0.5) + 0.15 * self._snap_ema[1],
        )

    @property
    def box_ints(self):
        x1, y1, x2, y2 = self._box
        return (int(x1), int(y1), int(x2), int(y2))

    def to_detection(self, screen_cx, screen_cy):
        x1, y1, x2, y2 = self.box_ints
        sx, sy = self._snap_ema
        snap_dist = ((sx - screen_cx) ** 2 + (sy - screen_cy) ** 2) ** 0.5
        return Detection(
            x1, y1, x2, y2, self.conf,
            track_id=self.id,
            _screen_cx=screen_cx, _screen_cy=screen_cy,
            _snap_dist=snap_dist,
        )


# ── 匹配工具函数 ────────────────────────────────────────────

def _iou_matrix(tracks, dets):
    n, m = len(tracks), len(dets)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)
    tb = np.array([t._box for t in tracks], dtype=np.float32)
    db = np.array([(d.x1, d.y1, d.x2, d.y2) for d in dets], dtype=np.float32)
    ix1 = np.maximum(tb[:, 0:1], db[:, 0])
    iy1 = np.maximum(tb[:, 1:2], db[:, 1])
    ix2 = np.minimum(tb[:, 2:3], db[:, 2])
    iy2 = np.minimum(tb[:, 3:4], db[:, 3])
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    area_t = (tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])
    area_d = (db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1])
    union = area_t[:, None] + area_d[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _snap_dist_matrix(tracks, dets):
    n, m = len(tracks), len(dets)
    if n == 0 or m == 0:
        return np.full((n, m), np.inf, dtype=np.float32)

    ts = np.array([t._snap_ema for t in tracks], dtype=np.float32)

    def _det_snap(d):
        head_h = max((d.y2 - d.y1) * HEAD_ZONE_RATIO, 10.0)
        return (d.x1 + d.x2) * 0.5, d.y1 + head_h * 0.5

    ds = np.array([_det_snap(d) for d in dets], dtype=np.float32)
    diff = ts[:, None, :] - ds[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def _greedy_match(iou, thresh):
    n_t, n_d = iou.shape
    used_t, used_d = set(), set()
    pairs = []
    flat = np.argsort(iou.flatten())[::-1]
    for idx in flat:
        ti, di = divmod(int(idx), n_d)
        if iou[ti, di] < thresh:
            break
        if ti not in used_t and di not in used_d:
            pairs.append((ti, di))
            used_t.add(ti)
            used_d.add(di)
    unmatched_t = [i for i in range(n_t) if i not in used_t]
    unmatched_d = [i for i in range(n_d) if i not in used_d]
    return pairs, unmatched_t, unmatched_d


def _greedy_match_by_dist(dist, max_dist):
    n_t, n_d = dist.shape
    used_t, used_d = set(), set()
    pairs = []
    flat = np.argsort(dist.flatten())
    for idx in flat:
        ti, di = divmod(int(idx), n_d)
        if dist[ti, di] > max_dist:
            break
        if ti not in used_t and di not in used_d:
            pairs.append((ti, di))
            used_t.add(ti)
            used_d.add(di)
    unmatched_t = [i for i in range(n_t) if i not in used_t]
    unmatched_d = [i for i in range(n_d) if i not in used_d]
    return pairs, unmatched_t, unmatched_d


# ── ByteTracker ──────────────────────────────────────────────

class ByteTracker:
    def __init__(self):
        self._tracks = []
        self._dead_snaps = []
        self._frame_count = 0

    def reset(self):
        self._tracks.clear()
        self._dead_snaps.clear()
        self._frame_count = 0
        _Track._id_counter = 1

    def update(self, detections, screen_w, screen_h):
        self._frame_count += 1

        for t in self._tracks:
            t.predict()

        high_dets = [d for d in detections if d.confidence >= TRACKER_HIGH_CONF]
        low_dets = [d for d in detections
                    if CONFIDENCE_THRESHOLD <= d.confidence < TRACKER_HIGH_CONF]

        # Stage 1: 高置信度匹配
        if self._tracks and high_dets:
            iou1 = _iou_matrix(self._tracks, high_dets)
            pairs1, unmatched_t1, unmatched_hd = _greedy_match(iou1, TRACKER_IOU_THRESH)
            for ti, di in pairs1:
                self._tracks[ti].update(high_dets[di])
        else:
            unmatched_t1 = list(range(len(self._tracks)))
            unmatched_hd = list(range(len(high_dets)))

        # Stage 1b: 快照距离回退匹配
        if unmatched_t1 and unmatched_hd:
            conf_unm_ti = [ti for ti in unmatched_t1
                           if self._tracks[ti].state == _CONFIRMED]
            if conf_unm_ti:
                sub_tracks = [self._tracks[ti] for ti in conf_unm_ti]
                sub_dets = [high_dets[di] for di in unmatched_hd]
                sdist = _snap_dist_matrix(sub_tracks, sub_dets)
                pairs1b, _, _ = _greedy_match_by_dist(sdist, TRACKER_CENTER_DIST_FALLBACK)
                matched_hd_1b = set()
                for subi, subdi in pairs1b:
                    orig_ti = conf_unm_ti[subi]
                    orig_di = unmatched_hd[subdi]
                    self._tracks[orig_ti].update(high_dets[orig_di])
                    matched_hd_1b.add(orig_di)
                    unmatched_t1.remove(orig_ti)
                if matched_hd_1b:
                    unmatched_hd = [di for di in unmatched_hd if di not in matched_hd_1b]

        # Stage 2: 低置信度匹配
        still_unmatched_t = set(unmatched_t1)
        confirmed_remaining = [
            (ti, self._tracks[ti])
            for ti in unmatched_t1
            if self._tracks[ti].state == _CONFIRMED
        ]
        if confirmed_remaining and low_dets:
            rc_tracks = [t for _, t in confirmed_remaining]
            rc_idx = [i for i, _ in confirmed_remaining]
            sdist2 = _snap_dist_matrix(rc_tracks, low_dets)
            pairs2, _, _ = _greedy_match_by_dist(sdist2, TRACKER_CENTER_DIST_FALLBACK)
            for ti2, di2 in pairs2:
                orig_ti = rc_idx[ti2]
                self._tracks[orig_ti].update_weak(low_dets[di2])
                still_unmatched_t.discard(orig_ti)

        # 创建新追踪
        def _det_snap(d):
            head_h = max((d.y2 - d.y1) * HEAD_ZONE_RATIO, 10.0)
            return (d.x1 + d.x2) * 0.5, d.y1 + head_h * 0.5

        active_snaps = [t._snap_ema for t in self._tracks]
        dedup_sq = TRACKER_DEDUP_DIST ** 2
        reid_sq = TRACKER_REID_DIST ** 2

        for di in unmatched_hd:
            if high_dets[di].confidence < NEW_TRACK_CONF:
                continue
            det = high_dets[di]
            dsnx, dsny = _det_snap(det)
            too_close = any(
                (sx - dsnx) ** 2 + (sy - dsny) ** 2 < dedup_sq
                for sx, sy in active_snaps
            )
            if too_close:
                continue

            det_cx = (det.x1 + det.x2) * 0.5
            det_cy = (det.y1 + det.y2) * 0.5
            reuse_id = None
            for idx, (sx, sy, old_id, _exp) in enumerate(self._dead_snaps):
                if (sx - det_cx) ** 2 + (sy - det_cy) ** 2 < reid_sq:
                    reuse_id = old_id
                    self._dead_snaps.pop(idx)
                    break

            new_track = _Track(det, reuse_id=reuse_id)
            self._tracks.append(new_track)
            active_snaps.append((dsnx, dsny))

        # 清理死追踪
        any_recently_detected = any(
            t.state == _CONFIRMED and t.miss_streak == 0
            for t in self._tracks
        )

        def _effective_max_age(t):
            if t.state != _CONFIRMED:
                return TRACKER_MAX_AGE
            if any_recently_detected and t.miss_streak < TRACKER_GHOST_MISS_LIMIT:
                return TRACKER_MAX_AGE * 10
            return TRACKER_MAX_AGE

        dead = [
            t for t in self._tracks
            if (t.state == _TENTATIVE and t.miss_streak >= TRACKER_MAX_AGE)
            or (t.state == _CONFIRMED and t.miss_streak >= _effective_max_age(t))
        ]
        for dt in dead:
            if dt.state == _CONFIRMED:
                self._dead_snaps.append(
                    (float(dt.kf.x[0]), float(dt.kf.x[1]), dt.id,
                     self._frame_count + TRACKER_REID_TTL)
                )

        self._dead_snaps = [s for s in self._dead_snaps if s[3] > self._frame_count]
        self._tracks = [t for t in self._tracks if t not in dead]

        cx, cy = screen_w // 2, screen_h // 2
        result = [t.to_detection(cx, cy)
                  for t in self._tracks if t.state == _CONFIRMED]
        result.sort(key=lambda d: d.distance_to_center)
        return result
