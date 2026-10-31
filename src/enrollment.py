"""Guided, user-confirmed poses; never silently attach another person's body."""
from __future__ import annotations

import math
import json
import os
import numpy as np


STEPS = (
    ("face_front", "FACE: look straight at the camera", 4),
    ("face_left", "FACE: turn your head slightly left; keep both eyes visible", 2),
    ("face_right", "FACE: turn your head slightly right; keep both eyes visible", 2),
    ("front", "BODY: stand facing the camera; show head to feet", 3),
    ("left", "BODY: slowly turn to your left side; keep your whole body visible", 3),
    ("right", "BODY: slowly turn to your right side; keep your whole body visible", 3),
    ("back", "BODY: slowly turn around; show your back to the camera", 3),
    ("diagonal", "BODY: turn to a diagonal view; keep your whole body visible", 3),
)


class GuidedEnrollment:
    def __init__(self, track_id, name, framing="upper"):
        self.track_id, self.name = track_id, name
        self.framing=framing
        self.step = 0
        self.faces, self.bodies, self.views = [], [], []
        self.count = 0
        self.capture_at = None
        self.last_sample = -1e6
        self.warning = ""
        self.failed = False
        self.last_seen = None
        self.paused = False
        self.resume_capture = False
        self.stable_frames = 0
        self.mismatch_frames = 0
        self.recovery_candidate = None
        self.recovery_hits = 0
        self.recovery_at = -1e6
        self.face_anchors = []
        self.anchor_features = []
        self.last_box = None

    def checkpoint(self,path):
        """Persist only enrollment data; live track IDs never survive a restart."""
        path.parent.mkdir(parents=True,exist_ok=True)
        metadata={"name":self.name,"step":self.step,"count":self.count,
                  "framing":self.framing,
                  "views":self.views,"capturing":self.capture_at is not None or self.resume_capture}
        temporary=path.with_suffix('.tmp.npz')
        np.savez(temporary,metadata=np.asarray(json.dumps(metadata)),
                 faces=np.asarray(self.faces,np.float32).reshape(-1,512),
                 bodies=np.asarray(self.bodies,np.float32).reshape(-1,512),
                 anchors=np.asarray(self.face_anchors,np.float32).reshape(-1,512))
        os.replace(temporary,path)

    @classmethod
    def restore(cls,path):
        with np.load(path,allow_pickle=False) as data:
            metadata=json.loads(str(data['metadata'].item()))
            wizard=cls(-1,metadata['name'],metadata.get('framing','upper'))
            wizard.step=int(metadata['step']);wizard.count=int(metadata['count'])
            wizard.faces=[v.copy() for v in data['faces']]
            wizard.bodies=[v.copy() for v in data['bodies']]
            wizard.face_anchors=[v.copy() for v in data['anchors']]
            wizard.views=list(metadata['views'])
        if (not 0<=wizard.step<=len(STEPS) or len(wizard.views)!=len(wizard.bodies)
                or not 0<=wizard.count<=4):
            raise ValueError('Invalid saved enrollment progress')
        for vector in wizard.faces+wizard.bodies+wizard.face_anchors:
            if vector.shape!=(512,) or not np.isfinite(vector).all():
                raise ValueError('Invalid saved enrollment embedding')
        wizard.paused=True
        wizard.resume_capture=bool(metadata['capturing'])
        wizard.warning='Draft restored. Look straight at the camera to recover your enrollment.'
        return wizard

    def anchor(self, track):
        """Remember the selected person before any guided samples exist."""
        if track is None:
            return
        if hasattr(track,"box"):
            self.last_box=track.box.copy()
        self.last_seen=getattr(track,"last_seen",self.last_seen)
        self.anchor_features=[v.copy() for v in getattr(track,"features",[])]
        if track.face is not None and track.face.vector is not None:
            self.face_anchors=[track.face.vector.copy()]

    @property
    def complete(self):
        return self.step == len(STEPS)

    def ready(self, now):
        if not self.complete and not self.failed:
            self.capture_at = now + 3.
            self.warning = ""

    def pause(self, reason):
        self.resume_capture = self.resume_capture or self.capture_at is not None
        self.capture_at = None
        self.paused = True
        self.stable_frames = 0
        self.warning = reason

    def recover_track(self, tracks, now):
        """Rebind a lost enrollment only after repeated matching face evidence."""
        current = next((t for t in tracks if t.track_id==self.track_id),None)
        references=self.faces+self.face_anchors
        if current is not None and current.visible or self.failed:
            self.recovery_candidate,self.recovery_hits = None,0
            return
        candidates=[]
        for track in tracks:
            if not track.visible or track.uncertain or track.face is None or track.face.vector is None:
                continue
            if not references:
                continue
            score=max(float(v @ track.face.vector) for v in references)
            candidates.append((score,track.track_id))
        candidates.sort(reverse=True)
        if not references:
            # Before the first usable face, only a short, unambiguous body
            # continuation at the same location can bootstrap recovery.
            from .person_tracking import iou,center_distance
            visible=[t for t in tracks if t.visible and not t.uncertain]
            if (len(visible)==1 and self.last_box is not None and self.last_seen is not None
                    and now-self.last_seen<=2.5 and self.anchor_features):
                other=visible[0]
                score=max((float(a @ b) for a in self.anchor_features
                           for b in getattr(other,"features",[])),default=0.)
                if (score>=.80 and iou(self.last_box,other.box)>=.5
                        and center_distance(self.last_box,other.box)<.2):
                    candidates=[(score,other.track_id)]
        if (not candidates or candidates[0][0]<.62
                or len(candidates)>1 and candidates[0][0]-candidates[1][0]<.06):
            self.recovery_candidate,self.recovery_hits = None,0
            return
        tid=candidates[0][1]
        if self.recovery_candidate==tid and now-self.recovery_at<=1.5:
            self.recovery_hits+=1
        else:
            self.recovery_candidate,self.recovery_hits=tid,1
        self.recovery_at=now
        if self.recovery_hits>=3:
            self.track_id=tid
            self.stable_frames=2
            self.warning="Matching face recovered. Resuming this person's enrollment."
            self.recovery_candidate,self.recovery_hits=None,0

    def snapshot(self, now):
        if self.failed:
            return {"name":self.name,"track_id":self.track_id,"instruction":"Face mismatch. Cancel and re-enroll from your face.",
                    "detail":self.warning,"progress":self.step/len(STEPS),"complete":False,"countdown":None}
        if self.paused:
            return {"name":self.name,"track_id":self.track_id,
                    "instruction":"Paused: enrollment stays locked and all captured samples are kept.",
                    "detail":self.warning,"progress":self.step/len(STEPS),
                    "complete":self.complete,"countdown":None}
        if self.complete:
            return {"name":self.name,"track_id":self.track_id,"instruction":"All views captured. Face the camera again, then click Save profile.",
                    "detail":"8 face samples + 15 body samples across 5 views", "progress":1.,
                    "complete":True,"countdown":None}
        key,instruction,needed=STEPS[self.step]
        if self.framing=='upper' and not key.startswith('face_'):
            instruction={
                'front':'UPPER BODY: face forward; show your head, shoulders and chest',
                'left':'UPPER BODY: slowly turn left; keep head and shoulders visible',
                'right':'UPPER BODY: slowly turn right; keep head and shoulders visible',
                'back':'UPPER BODY: turn around; show the back of your head and shoulders',
                'diagonal':'UPPER BODY: take a diagonal view; show head, shoulders and chest',
            }[key]
        remaining = max(0, math.ceil(self.capture_at-now)) if self.capture_at is not None else None
        detail = self.warning or (f"Hold this pose: {self.count}/{needed} samples" if remaining==0 else
                                  f"Capture in {remaining}" if remaining is not None else
                                  "Take the shown pose, then click Ready / press Space")
        return {"name":self.name,"track_id":self.track_id,"instruction":instruction,"detail":detail,
                "progress":(self.step+self.count/needed)/len(STEPS),"complete":False,
                "countdown":remaining,"step":key}

    def update(self, track, observation, now, frame_shape, others_visible=0):
        if self.failed:
            return
        if track is None or not track.visible:
            self.pause("Target not visible. Samples kept; face the camera to recover the locked person.")
            return
        self.last_seen=now
        # Unrelated people may remain visible. Only interference with this
        # locked target pauses enrollment; it never changes the selected ID.
        if track.uncertain or observation is None or observation.occluded:
            self.pause("Locked target is obscured or ambiguous. Samples kept; separate and face the camera.")
            return
        face=track.face
        references=self.faces+self.face_anchors
        face_step=not self.complete and STEPS[self.step][0].startswith('face_')
        # During body poses, low similarity from a distant/profile face is not
        # evidence of a switch. Only a positively recognized conflicting face
        # interrupts body sampling; a clear face is required again at Save.
        conflicting_name=(face is not None and getattr(face,'name',None) is not None
                          and face.name!=self.name)
        if face is not None and face.vector is not None and references and (face_step or conflicting_name):
            if max(float(v @ face.vector) for v in references)<.45:
                self.mismatch_frames+=1
                self.pause("Face check differs. Samples kept; look straight at the camera.")
                return
            self.mismatch_frames=0
        if face is not None and face.vector is not None and not self.face_anchors:
            self.face_anchors=[face.vector.copy()]
        if hasattr(track,"box"):
            self.last_box=track.box.copy()
        features=getattr(track,"features",[])
        if features:
            self.anchor_features=[v.copy() for v in list(features)[-8:]]
        if self.paused:
            self.stable_frames+=1
            if self.stable_frames<3:
                self.warning="Locked target recovered. Checking stability before resuming..."
                return
            self.paused=False
            self.warning=""
            if self.resume_capture:
                self.capture_at=now+1.
            self.resume_capture=False
        if self.complete:
            return
        if self.capture_at is None or now<self.capture_at or now-self.last_sample<.4:
            return
        key,_,needed=STEPS[self.step]
        if key.startswith("face_"):
            if face is None or face.vector is None:
                self.warning="Need a clear face: move closer, keep both eyes visible, improve lighting"
                return
            self.faces.append(face.vector.copy())
        else:
            h,w=frame_shape[:2]
            x1,y1,x2,y2=observation.box
            if self.framing=='full' and (x1<=w*.01 or x2>=w*.99 or y1<=h*.01 or y2>=h*.99):
                self.warning="Step back until head and feet fit inside the image; avoid frame edges"
                return
            minimum_height=max(160,h*.25) if self.framing=='full' else max(120,h*.20)
            if y2-y1<minimum_height or x2-x1<48:
                self.warning="Move a little closer: the detected body is too small for a useful sample"
                return
            if observation.feature is None:
                self.warning="Waiting for a clear body detection"
                return
            self.bodies.append(observation.feature.copy())
            self.views.append(key)
        self.last_sample=now
        self.warning=""
        self.count+=1
        if self.count>=needed:
            self.step+=1
            self.count=0
            self.capture_at=None

    def save_allowed(self, track):
        if not self.complete or self.failed or self.paused:
            return False
        if track is None or not track.visible or track.uncertain or track.face is None or track.face.vector is None:
            return False
        return max(float(v @ track.face.vector) for v in self.faces)>=.62
