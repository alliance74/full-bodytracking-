"""Motion + learned appearance association with conservative identity memory.

This tracker keeps a name only after repeated face evidence. Body embeddings
maintain continuity; they do not turn an unrecognized stranger into an employee.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    inter = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
    union = np.maximum(a[2:] - a[:2], 0).prod() + np.maximum(b[2:] - b[:2], 0).prod() - inter
    return float(inter / max(float(union), 1e-8))


def center_distance(a, b):
    diagonal = max(float(np.linalg.norm(a[2:] - a[:2])), 1.)
    return float(np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2) / diagonal)


@dataclass
class Observation:
    box: np.ndarray
    score: float
    feature: np.ndarray | None
    occluded: bool = False


@dataclass
class PersonTrack:
    track_id: int
    box: np.ndarray
    created: float
    last_seen: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))
    features: deque = field(default_factory=lambda: deque(maxlen=24))
    hits: int = 1
    clean_hits: int = 1
    name: str | None = None
    candidate: str | None = None
    candidate_hits: int = 0
    candidate_at: float = -1e6
    last_face: float = -1e6
    identity_source: str = "face"
    body_identity_at: float = -1e6
    body_candidate: str | None = None
    body_hits: int = 0
    body_candidate_at: float = -1e6
    body_score: float | None = None
    uncertain: bool = False
    reacquired: bool = False
    visible: bool = True
    last_sample: float = -1e6
    face: object = None
    trail: deque = field(default_factory=lambda: deque(maxlen=24))

    def prediction(self, now):
        # Predict translation only: turning changes box width and should not
        # extrapolate an increasingly narrow or inverted bounding box.
        shift=(self.velocity[:2]+self.velocity[2:])/2
        return self.box + np.tile(shift,2) * min(max(now-self.last_seen,0),.75)

    def similarity(self, feature):
        if feature is None or not self.features:
            return None
        return float(max(np.dot(v, feature) for v in self.features))

    def evidence(self, now, face_hold=60.):
        if not self.visible:
            return "lost"
        if self.uncertain:
            return "uncertain"
        evidence_at = self.last_face if self.identity_source == "face" else self.body_identity_at
        if self.name is None or now - evidence_at > face_hold:
            return "unknown"
        if self.identity_source == "body gallery":
            return "body gallery (unverified)"
        if (now - self.last_face < .8 and self.face is not None
                and self.face.name == self.name and self.face.vector is not None):
            return "face confirmed"
        return "body continuity" if not self.reacquired else "body reacquired"

    def display_name(self, now, face_hold=60.):
        return self.name if self.evidence(now, face_hold) in (
            "face confirmed", "body continuity", "body reacquired", "body gallery (unverified)") else None


class PersonTracker:
    def __init__(self, max_lost=8., face_hold=60., high_threshold=.35):
        self.max_lost = max_lost
        self.face_hold = face_hold
        self.high_threshold = high_threshold
        self.tracks: dict[int, PersonTrack] = {}
        self.next_id = 1

    def _associate(self, tracks, observations, indices, now, low=False):
        if not tracks or not indices:
            return [], set(), set()
        costs = np.full((len(tracks), len(indices)), 1000., np.float32)
        for row, track in enumerate(tracks):
            elapsed = now - track.last_seen
            for col, index in enumerate(indices):
                obs = observations[index]
                overlap = max(iou(track.prediction(now), obs.box),iou(track.box,obs.box))
                distance = min(center_distance(track.prediction(now),obs.box),center_distance(track.box,obs.box))
                similarity = track.similarity(obs.feature)
                if low:
                    if elapsed > 1.5 or overlap < .25:
                        continue
                    costs[row, col] = 1 - overlap
                elif elapsed > 2.5:
                    # A returning body needs strong appearance and plausible position.
                    if similarity is None or similarity < .80 or distance > 2.:
                        continue
                    costs[row, col] = .8 * (1 - similarity) + .2 * min(distance, 1)
                else:
                    if overlap < .05 and distance > .65:
                        continue
                    if similarity is not None and similarity < .35:
                        continue
                    appearance_cost = 1 - similarity if similarity is not None else .5
                    costs[row, col] = .5 * (1 - overlap) + .35 * appearance_cost + .15 * min(distance, 1)
        rows, cols = linear_sum_assignment(costs)
        matches, used_tracks, used_indices = [], set(), set()
        for row, col in zip(rows, cols):
            if costs[row, col] >= (0.78 if low else .82):
                continue
            track = tracks[row]
            index = indices[col]
            other_tracks = np.delete(costs[:, col], row)
            other_boxes = np.delete(costs[row], col)
            margin = min(float(other_tracks.min()) if len(other_tracks) else 1000.,
                         float(other_boxes.min()) if len(other_boxes) else 1000.) - float(costs[row, col])
            ambiguous = not low and margin < .08
            if now - track.last_seen > 1. and ambiguous:
                # Similar uniforms returning together: do not guess which name belongs where.
                track.uncertain = True
                continue
            matches.append((track, index, ambiguous))
            used_tracks.add(track.track_id)
            used_indices.add(index)
        return matches, used_tracks, used_indices

    def update(self, observations, now):
        for tid, track in list(self.tracks.items()):
            track.face = None
            if now - track.last_seen > self.max_lost:
                del self.tracks[tid]
        high = [i for i, obs in enumerate(observations) if obs.score >= self.high_threshold]
        low = [i for i, obs in enumerate(observations) if obs.score < self.high_threshold]
        tracks = list(self.tracks.values())
        # Associate live tracks first. Similar old fragments of this same
        # person must not compete with a healthy currently visible track.
        active=[t for t in tracks if t.visible and now-t.last_seen<=1.5]
        dormant=[t for t in tracks if t not in active]
        matches, used_tracks, used = self._associate(active, observations, high, now)
        remaining_high=[i for i in high if i not in used]
        recovered,recovered_tracks,recovered_indices=self._associate(dormant,observations,remaining_high,now)
        matches+=recovered
        used_tracks|=recovered_tracks
        used|=recovered_indices
        remaining = [t for t in tracks if t.track_id not in used_tracks]
        low_matches, low_used, _ = self._associate(remaining, observations, low, now, low=True)
        used_tracks |= low_used
        output = {}
        for track, index, ambiguous in matches + low_matches:
            obs = observations[index]
            dt = max(now - track.last_seen, .03)
            if dt > 1.:
                track.reacquired = True
                track.velocity[:] = 0
                track.candidate, track.candidate_hits = None, 0
            else:
                center_delta=(obs.box[:2]+obs.box[2:]-track.box[:2]-track.box[2:])/2
                limit=max(float(np.linalg.norm(track.box[2:]-track.box[:2])),1.)
                velocity = np.tile(np.clip(center_delta/dt,-limit,limit),2)
                track.velocity = .5 * track.velocity + .5 * velocity
            track.box = obs.box.copy()
            track.last_seen = now
            track.hits += 1
            track.visible = True
            if ambiguous or obs.occluded:
                track.clean_hits = 0
                track.uncertain = True
                track.candidate, track.candidate_hits = None, 0
            elif obs.score >= self.high_threshold:
                track.clean_hits += 1
                # An unknown track may become safe to enroll after a crossing.
                # A named track still requires its face to resolve uncertainty.
                if track.name is None and track.clean_hits >= 3:
                    track.uncertain = False
            # Avoid learning a crossing person's appearance into the track gallery.
            if (not track.uncertain and not obs.occluded and obs.feature is not None
                    and obs.score >= self.high_threshold and now - track.last_sample >= .4):
                track.features.append(obs.feature.copy())
                track.last_sample = now
            track.trail.append(tuple(((track.box[:2] + track.box[2:]) / 2).astype(int)))
            output[index] = track
        for track in tracks:
            if track.track_id not in used_tracks:
                track.visible = False
                track.candidate, track.candidate_hits = None, 0
        for index in high:
            if index in used:
                continue
            obs = observations[index]
            track = PersonTrack(self.next_id, obs.box.copy(), now, now)
            track.uncertain = obs.occluded
            track.clean_hits = 0 if obs.occluded else 1
            if obs.feature is not None and not obs.occluded:
                track.features.append(obs.feature.copy())
                track.last_sample = now
            self.next_id += 1
            self.tracks[track.track_id] = track
            output[index] = track
        return output

    def observe_face(self, track, name, now):
        if name is None:
            track.candidate, track.candidate_hits = None, 0
            return
        if track.name is not None and track.name != name:
            # A strong contradictory face must stop the old identity immediately.
            track.name = None
            track.last_face = -1e6
            track.uncertain = True
            track.features.clear()
            track.body_candidate, track.body_hits = None, 0
        if any(t is not track and t.visible and t.name == name and not t.uncertain
               for t in self.tracks.values()):
            track.uncertain = True
            track.candidate, track.candidate_hits = None, 0
            return
        if track.candidate == name and now - track.candidate_at <= 1.5:
            track.candidate_hits += 1
        else:
            track.candidate, track.candidate_hits = name, 1
        track.candidate_at = now
        if track.candidate_hits >= 3:
            track.name = name
            track.last_face = now
            track.identity_source = "face"
            track.uncertain = False
            track.reacquired = False

    def observe_body(self, track, name, score, now):
        """Stored appearance proposes an identity, never a face confirmation."""
        if track.uncertain or track.clean_hits < 3:
            track.body_candidate, track.body_hits = None, 0
            return
        if name is None:
            track.body_candidate, track.body_hits = None, 0
            return
        if track.name is not None and track.identity_source == "face":
            return
        if track.name is not None and track.name != name:
            track.uncertain = True
            return
        if any(t is not track and t.visible and t.name == name and not t.uncertain
               for t in self.tracks.values()):
            track.body_candidate, track.body_hits = None, 0
            return
        if track.body_candidate == name and now-track.body_candidate_at<=1.5:
            track.body_hits += 1
        else:
            track.body_candidate, track.body_hits = name, 1
        track.body_candidate_at = now
        track.body_score = score
        if track.body_hits>=3:
            track.name = name
            track.identity_source = "body gallery"
            track.body_identity_at = now


def associate_faces(faces, person_boxes):
    """One face to one body; reject ambiguous overlapping head regions."""
    candidates = []
    for fi, face in enumerate(faces):
        x1, y1, x2, y2 = face.box
        center_x, center_y = (x1+x2)/2, (y1+y2)/2
        area = max(float((x2-x1)*(y2-y1)), 1.)
        possible = []
        for pi, box in enumerate(person_boxes):
            bx1, by1, bx2, by2 = box
            if not (bx1 <= center_x <= bx2 and by1 <= center_y <= by1 + .55*(by2-by1)):
                continue
            intersection = max(0, min(x2,bx2)-max(x1,bx1)) * max(0,min(y2,by2)-max(y1,by1))
            if intersection / area < .8:
                continue
            horizontal = abs(center_x-(bx1+bx2)/2) / max(bx2-bx1, 1)
            possible.append((float(horizontal), pi))
        possible.sort()
        if not possible or len(possible) > 1 and possible[1][0]-possible[0][0] < .15:
            continue
        candidates.append((possible[0][0], fi, possible[0][1]))
    used_faces, used_people, assignments = set(), set(), {}
    for _, fi, pi in sorted(candidates):
        if fi not in used_faces and pi not in used_people:
            assignments[pi] = fi
            used_faces.add(fi)
            used_people.add(pi)
    return assignments
