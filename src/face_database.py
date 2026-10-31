"""The previous project's ArcFace enrollment format, with ambiguity checks."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BODY_PREFIX = "__body__::"
VIEW_PREFIX = "__views__::"
BODY_VIEWS = ("front", "left", "right", "back", "diagonal")


class FaceDatabase:
    def __init__(self, path=None, threshold=.38, margin=.06):
        self.path = Path(path or ROOT / "data/db/face_db.npz")
        self.threshold = threshold
        self.margin = margin
        self.reload()

    def reload(self):
        self.entries = {}
        self.body_entries = {}
        self.body_views = {}
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as data:
                for name in data.files:
                    if name.startswith((BODY_PREFIX, VIEW_PREFIX)):
                        continue
                    vectors = np.asarray(data[name], np.float32).reshape(-1, 512)
                    vectors = vectors[np.isfinite(vectors).all(axis=1)]
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                    valid = norms[:, 0] > 1e-8
                    if valid.any():
                        self.entries[name] = vectors[valid] / norms[valid]
                for name in self.entries:
                    if BODY_PREFIX+name not in data.files or VIEW_PREFIX+name not in data.files:
                        continue
                    vectors = np.asarray(data[BODY_PREFIX+name],np.float32).reshape(-1,512)
                    views = data[VIEW_PREFIX+name].astype(str).tolist()
                    norms = np.linalg.norm(vectors,axis=1,keepdims=True)
                    if (len(views)==len(vectors) and set(views)==set(BODY_VIEWS)
                            and np.isfinite(vectors).all() and (norms>1e-8).all()):
                        self.body_entries[name] = vectors/norms
                        self.body_views[name] = views

    def match(self, vector):
        if not self.entries:
            return None, None
        ranked = sorted((1 - float(np.max(values @ vector)), name)
                        for name, values in self.entries.items())
        distance, name = ranked[0]
        if distance > self.threshold:
            return None, distance
        if len(ranked) > 1 and ranked[1][0] - distance < self.margin:
            return None, distance
        return name, distance

    def match_body(self, vector, threshold=.90, margin=.10):
        if vector is None or not self.body_entries:
            return None, None, None
        ranked = sorted(((float(np.max(values @ vector)), name)
                         for name, values in self.body_entries.items()), reverse=True)
        score, name = ranked[0]
        gap = score-ranked[1][0] if len(ranked)>1 else 1.
        return (name if score>=threshold and gap>=margin else None), score, gap

    def enroll(self, name, vectors, bodies=None, views=None):
        name = name.strip()
        if (not name or len(name) > 80 or any(ord(c) < 32 for c in name)
                or name.startswith((BODY_PREFIX,VIEW_PREFIX))):
            raise ValueError("Enter a name of 1–80 printable characters")
        values = np.asarray(vectors, np.float32).reshape(-1, 512)
        if len(values) < 5 or not np.isfinite(values).all():
            raise ValueError("At least five valid face samples are required")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if (norms < 1e-8).any():
            raise ValueError("Face samples must have nonzero embeddings")
        values /= norms
        entries = dict(self.entries)
        previous = entries.get(name, np.empty((0, 512), np.float32))
        entries[name] = np.concatenate((previous, values))[-40:]
        body_entries = dict(self.body_entries)
        body_views = dict(self.body_views)
        if bodies is not None or views is not None:
            body_values = np.asarray(bodies,np.float32).reshape(-1,512)
            labels = list(views or [])
            body_norms = np.linalg.norm(body_values,axis=1,keepdims=True)
            if (len(labels)!=len(body_values) or set(labels)!=set(BODY_VIEWS)
                    or any(labels.count(view)<3 for view in BODY_VIEWS)
                    or not np.isfinite(body_values).all() or (body_norms<1e-8).any()):
                raise ValueError("Capture three valid body samples for each of the five guided views")
            body_entries[name] = body_values/body_norms
            body_views[name] = labels
            # Re-registration replaces the face profile along with its body views.
            entries[name] = values[-40:]
        payload = dict(entries)
        for body_name, body_values in body_entries.items():
            payload[BODY_PREFIX+body_name] = body_values
            payload[VIEW_PREFIX+body_name] = np.asarray(body_views[body_name],dtype="U12")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp.npz")
        np.savez(temporary, **payload)
        os.replace(temporary, self.path)
        self.entries = entries
        self.body_entries, self.body_views = body_entries, body_views
