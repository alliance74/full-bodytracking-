"""Lightweight IoU face tracker and identity lock state for the live demo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Track:
    track_id: int
    box: tuple[int, int, int, int]
    missed: int = 0


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(1, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union


class IoUTracker:
    """Assigns stable IDs to detections across frames using greedy IoU matching."""
    def __init__(self, max_missed: int = 12, min_iou: float = 0.12):
        self.max_missed = max_missed
        self.min_iou = min_iou
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def update(self, boxes) -> list[int]:
        pairs = sorted(
            ((_iou(track.box, box), tid, i) for tid, track in self.tracks.items()
             for i, box in enumerate(boxes)), reverse=True
        )
        used_tracks, used_boxes, assigned = set(), set(), {}
        for score, tid, i in pairs:
            if score < self.min_iou:
                break
            if tid in used_tracks or i in used_boxes:
                continue
            self.tracks[tid].box = tuple(map(int, boxes[i]))
            self.tracks[tid].missed = 0
            assigned[i] = tid
            used_tracks.add(tid)
            used_boxes.add(i)
        for tid, track in list(self.tracks.items()):
            if tid not in used_tracks:
                track.missed += 1
                if track.missed > self.max_missed:
                    del self.tracks[tid]
        for i, box in enumerate(boxes):
            if i not in assigned:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(tid, tuple(map(int, box)))
                assigned[i] = tid
        return [assigned[i] for i in range(len(boxes))]


class IdentityLock:
    """Keeps one selected track and recognized name as the active identity."""
    def __init__(self):
        self.track_id: Optional[int] = None
        self.name: Optional[str] = None

    @property
    def active(self) -> bool:
        return self.track_id is not None

    def lock(self, track_id: int, name: str) -> None:
        self.track_id = track_id
        self.name = name

    def rebind(self, track_id: int, recognized_name: str) -> bool:
        """Attach the existing identity lock to a newly assigned track ID."""
        if not self.active or recognized_name != self.name:
            return False
        self.track_id = track_id
        return True

    def unlock(self) -> None:
        self.track_id = None
        self.name = None

    def status(self, track_id: int, recognized_name: Optional[str]) -> str:
        if not self.active or track_id != self.track_id:
            return ""
        return "LOCKED" if recognized_name == self.name else "IDENTITY MISMATCH"
