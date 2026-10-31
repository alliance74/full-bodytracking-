"""Associate face identity with a person track, including face-hidden periods."""
from __future__ import annotations

from dataclasses import dataclass
from queue import SimpleQueue
import time
import numpy as np

from .face_database import FaceDatabase
from .enrollment import GuidedEnrollment
from .person_tracking import PersonTracker, Observation, associate_faces, iou
from .vision_models import (PersonDetector, FaceDetector, BodyEmbedder,
                            ArcFaceEmbedderONNX, align_face, face_quality, model_paths,appearance_box)


@dataclass
class FrameResult:
    frame: np.ndarray
    captured_at: float
    tracks: list[dict]
    faces: list
    enrollment_samples: int
    identities: int
    message: str
    elapsed_ms: float
    enrollment: dict | None = None
    thresholds: dict | None = None


class PersonPipeline:
    def __init__(self, models, database=None, face_threshold=.38, max_lost=8., face_hold=60.,
                 person_threshold=.35, max_people=12, body_threshold=.90, body_margin=.10,
                 body_framing="upper"):
        paths = model_paths(models)
        self.people = PersonDetector(paths["person"], threshold=person_threshold, max_people=max_people)
        self.faces = FaceDetector(paths["face"])
        self.arcface = ArcFaceEmbedderONNX(str(paths["arcface"]))
        self.body = BodyEmbedder(paths["body"])
        self.database = FaceDatabase(database, threshold=face_threshold)
        self.tracker = PersonTracker(max_lost, face_hold, person_threshold)
        self.commands = SimpleQueue()
        self.enrolling = None
        self.save_requested = False
        self.body_threshold, self.body_margin = body_threshold, body_margin
        self.body_framing=body_framing
        self.draft_path=self.database.path.with_name('enrollment_draft.npz')
        self.draft_signature=None
        self.message = "Click a person to select; face identity needs 3 clear observations"
        if self.draft_path.is_file():
            try:
                self.enrolling=GuidedEnrollment.restore(self.draft_path)
                self.body_framing=self.enrolling.framing
                self.message="Enrollment draft restored. Look at the camera to resume."
            except (OSError,ValueError,KeyError) as error:
                self.message=f"Could not restore draft: {error}"

    def _checkpoint(self):
        path=getattr(self,'draft_path',self.database.path.with_name('enrollment_draft.npz'))
        if self.enrolling is None:
            path.unlink(missing_ok=True)
            self.draft_signature=None
            return
        wizard=self.enrolling
        signature=(wizard.name,wizard.step,wizard.count,len(wizard.faces),len(wizard.bodies),len(wizard.face_anchors))
        if getattr(self,'draft_signature',None)!=signature:
            wizard.checkpoint(path)
            self.draft_signature=signature

    def _commands(self, now):
        while not self.commands.empty():
            command, payload = self.commands.get()
            try:
                if command == "enroll":
                    if self.enrolling is not None:
                        self.message="Enrollment is locked to one person; Cancel before enrolling someone else"
                        continue
                    tid, name = payload
                    if not name.strip() or len(name)>80:
                        raise ValueError("Enter a name of 1–80 characters")
                    self.enrolling = GuidedEnrollment(tid,name.strip(),getattr(self,'body_framing','upper'))
                    self.enrolling.anchor(self.tracker.tracks.get(tid))
                    self.message = "Enrollment target locked. Follow the prompts; avoid overlapping the target."
                elif command == "ready":
                    if self.enrolling is not None:
                        self.enrolling.ready(now)
                elif command == "save":
                    if self.enrolling is None:
                        self.message = "Press Enroll on a selected person first"
                        continue
                    self.save_requested = True
                elif command == "cancel":
                    self.enrolling = None
                    self.save_requested = False
                    self.message = "Enrollment cancelled"
                elif command == "reload":
                    self.database.reload()
                    self.enrolling = None
                    self.save_requested = False
                    # Existing identity evidence must be obtained again from this gallery.
                    for track in self.tracker.tracks.values():
                        track.name, track.candidate, track.candidate_hits = None, None, 0
                        track.last_face = -1e6
                    self.message = "Face database reloaded"
            except (ValueError, OSError) as error:
                self.message = f"Could not {command}: {error}"

    def process(self, frame, captured_at=None):
        start = time.monotonic()
        now = start if captured_at is None else captured_at
        self._commands(now)
        detected = self.people.detect(frame)
        observations = []
        crops, feature_indices = [], []
        for index, (box, score) in enumerate(detected):
            occluded = any(iou(box, other) > .25 for j, (other, _) in enumerate(detected) if j != index)
            x1, y1, x2, y2 = appearance_box(box,getattr(self,'body_framing','upper')).astype(int)
            crop = frame[y1:y2, x1:x2]
            observations.append(Observation(box, score, None, occluded))
            if score >= self.people.threshold and not occluded:
                crops.append(crop)
                feature_indices.append(index)
        for index, feature in zip(feature_indices, self.body.embed_many(crops)):
            observations[index].feature = feature
        assigned = self.tracker.update(observations, now)
        faces = self.faces.detect(frame)
        for face in faces:
            face.quality = face_quality(frame, face)
            if face.quality:
                aligned = align_face(frame, face.keypoints)
                if aligned is not None:
                    face.vector = self.arcface.embed(aligned)
                    face.name, face.distance = self.database.match(face.vector)
        associations = associate_faces(faces, [obs.box for obs in observations])
        # More than one simultaneous face claiming one enrolled name is not
        # enough evidence to choose which body owns that identity.
        names = [faces[fi].name for pi, fi in associations.items() if pi in assigned and faces[fi].name]
        duplicates = {name for name in names if names.count(name) > 1}
        for index, track in assigned.items():
            face = faces[associations[index]] if index in associations else None
            track.face = face
            track.body_score = None
            if observations[index].occluded or observations[index].score < self.people.threshold:
                continue
            if face is not None and face.name in duplicates:
                track.uncertain = True
                track.candidate, track.candidate_hits = None, 0
            else:
                self.tracker.observe_face(track, face.name if face is not None else None, now)
            # A clear but unrecognized face must not be replaced by a clothing
            # match. Stored body views are used only when the face is unavailable.
            if face is None or face.vector is None:
                candidate,score,gap = self.database.match_body(
                    observations[index].feature,self.body_threshold,self.body_margin)
                track.body_score = score
                if (score is not None and score>=self.body_threshold and gap<self.body_margin
                        and (track.name is None or track.identity_source=="body gallery")):
                    track.uncertain=True
                self.tracker.observe_body(track,candidate,score,now)
            else:
                track.body_candidate,track.body_hits = None,0
                if face.name is None and track.identity_source=="body gallery":
                    # A newly visible unknown face invalidates an outfit-only
                    # name rather than carrying it onto a lookalike person.
                    track.name=None
                    track.body_identity_at=-1e6
                    track.uncertain=True
        if self.enrolling is not None:
            wizard = self.enrolling
            wizard.recover_track(list(self.tracker.tracks.values()),now)
            selected_index = next((i for i,t in assigned.items() if t.track_id==wizard.track_id),None)
            selected_track = self.tracker.tracks.get(wizard.track_id)
            observation = observations[selected_index] if selected_index is not None else None
            others = sum(t.visible and t.track_id!=wizard.track_id for t in self.tracker.tracks.values())
            wizard.update(selected_track,observation,now,frame.shape,others)
            if self.save_requested:
                if wizard.save_allowed(selected_track):
                    try:
                        self.database.enroll(wizard.name,wizard.faces,wizard.bodies,wizard.views)
                        self.message = f"Saved face and five body views for {wizard.name}"
                        self.enrolling = None
                    except (ValueError,OSError) as error:
                        self.message = f"Could not save: {error}"
                else:
                    self.message = "Complete every view, then face the camera clearly again before Save"
                self.save_requested=False
        try:
            self._checkpoint()
        except OSError as error:
            self.message=f"Enrollment draft could not be saved: {error}"
        records = []
        for track in self.tracker.tracks.values():
            records.append({
                "track_id": track.track_id, "box": track.box.astype(int).tolist(),
                "name": track.display_name(now, self.tracker.face_hold),
                "evidence": track.evidence(now, self.tracker.face_hold),
                "visible": track.visible, "hits": track.hits,
                "face_visible": track.face is not None,
                "face_usable": bool(track.face is not None and track.face.vector is not None),
                "face_age": (round(max(0, now-track.last_face), 1)
                             if track.name and track.identity_source=="face" else None),
                "identity_source": track.identity_source,
                "body_similarity": track.body_score,
                "face_similarity": (max(0.,min(1.,1-track.face.distance)) if track.face is not None
                                    and track.face.distance is not None else None),
                "lost_seconds": round(max(0, now-track.last_seen), 1),
                "trail": list(track.trail),
            })
        return FrameResult(frame, now, records, faces,
                           len(self.enrolling.faces) if self.enrolling else 0,
                           len(self.database.entries),self.message,(time.monotonic()-start)*1000,
                           self.enrolling.snapshot(now) if self.enrolling else None,
                           {"face":1-self.database.threshold,"face_margin":self.database.margin,
                            "body":self.body_threshold,"body_margin":self.body_margin})
