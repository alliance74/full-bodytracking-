"""Deterministic identity tests; no model downloads or camera needed."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from src.face_database import FaceDatabase
from src.person_tracking import Observation, PersonTracker, PersonTrack, associate_faces


def feature(index):
    v=np.zeros(512,np.float32)
    v[index]=1
    return v


def obs(x=0,identity=0,score=.9,occluded=False):
    return Observation(np.array([x,0,x+80,240],np.float32),score,feature(identity),occluded)


class TrackingTests(unittest.TestCase):
    def confirm(self, tracker, track, name, now=0.):
        for offset in (0.,.2,.4):
            tracker.observe_face(track,name,now+offset)

    def test_turn_away_keeps_same_body_and_name(self):
        t=PersonTracker()
        track=t.update([obs()],0)[0]
        self.confirm(t,track,'Alice')
        original=track.track_id
        for n in range(1,15):
            track=t.update([obs(n*2)],n*.2+.4)[0]
            t.observe_face(track,None,n*.2+.4)
        self.assertEqual(track.track_id,original)
        self.assertEqual(track.display_name(3.3),'Alice')
        self.assertEqual(track.evidence(3.3),'body continuity')

    def test_three_face_observations_required(self):
        t=PersonTracker()
        track=t.update([obs()],0)[0]
        t.observe_face(track,'Alice',0)
        t.observe_face(track,'Alice',.2)
        self.assertIsNone(track.name)
        t.observe_face(track,'Alice',.4)
        self.assertEqual(track.name,'Alice')

    def test_interrupted_face_evidence_does_not_accumulate(self):
        t=PersonTracker()
        track=t.update([obs()],0)[0]
        t.observe_face(track,'Alice',0)
        t.observe_face(track,None,.2)
        t.observe_face(track,'Alice',.4)
        t.observe_face(track,'Alice',.6)
        self.assertIsNone(track.name)

    def test_conflicting_face_revokes_name_immediately(self):
        t=PersonTracker()
        track=t.update([obs()],0)[0]
        self.confirm(t,track,'Alice')
        t.observe_face(track,'Bob',.6)
        self.assertIsNone(track.name)
        self.assertTrue(track.uncertain)
        self.confirm(t,track,'Bob',.8)
        self.assertEqual(track.name,'Bob')

    def test_similar_uniform_overlap_hides_identity(self):
        t=PersonTracker()
        tracks=t.update([obs(0),obs(160)],0)
        self.confirm(t,tracks[0],'Alice')
        self.confirm(t,tracks[1],'Bob')
        result=t.update([obs(55,occluded=True),obs(100,occluded=True)],.6)
        self.assertTrue(all(track.uncertain for track in result.values()))
        self.assertTrue(all(track.display_name(.6) is None for track in result.values()))
        self.assertTrue(all(len(track.features)==1 for track in result.values()))

    def test_distinct_body_features_survive_detection_order_swap(self):
        t=PersonTracker()
        a=t.update([obs(0,0),obs(160,1)],0)
        aid,bid=a[0].track_id,a[1].track_id
        b=t.update([obs(155,1),obs(5,0)],.2)
        self.assertEqual(b[0].track_id,bid)
        self.assertEqual(b[1].track_id,aid)

    def test_out_of_view_is_lost_not_visible(self):
        t=PersonTracker()
        track=t.update([obs()],0)[0]
        self.confirm(t,track,'Alice')
        t.update([],1)
        self.assertFalse(track.visible)
        self.assertEqual(track.evidence(1),'lost')
        self.assertIsNone(track.display_name(1))
        t.update([],9)
        self.assertFalse(t.tracks)

    def test_reappearance_requires_body_match(self):
        t=PersonTracker()
        track=t.update([obs()],0)[0]
        self.confirm(t,track,'Alice')
        t.update([],1)
        result=t.update([obs(15)],2)[0]
        self.assertEqual(result.track_id,track.track_id)
        self.assertEqual(result.evidence(2),'body reacquired')
        self.assertEqual(result.display_name(2),'Alice')

    def test_lookalike_reappearance_does_not_guess(self):
        t=PersonTracker()
        tracks=t.update([obs(0),obs(100)],0)
        self.confirm(t,tracks[0],'Alice')
        self.confirm(t,tracks[1],'Bob')
        old_ids={v.track_id for v in tracks.values()}
        t.update([],1)
        results=t.update([obs(50)],2)
        self.assertNotIn(results[0].track_id,old_ids)
        self.assertIsNone(results[0].name)

    def test_unrelated_appearance_does_not_steal_track(self):
        t=PersonTracker()
        track=t.update([obs(0,0)],0)[0]
        self.confirm(t,track,'Alice')
        other=t.update([obs(0,1)],.6)[0]
        self.assertNotEqual(track.track_id,other.track_id)
        self.assertIsNone(other.name)

    def test_low_confidence_can_maintain_but_not_create_track(self):
        t=PersonTracker()
        self.assertFalse(t.update([obs(score=.2)],0))
        first=t.update([obs()],.2)[0]
        second=t.update([obs(2,score=.2)],.4)[0]
        self.assertEqual(first.track_id,second.track_id)
        self.assertEqual(len(second.features),1)

    def test_face_identity_expires_without_refresh(self):
        t=PersonTracker(face_hold=2)
        track=t.update([obs()],0)[0]
        self.confirm(t,track,'Alice')
        self.assertIsNone(track.display_name(3,2))
        self.assertEqual(track.evidence(3,2),'unknown')

    def test_same_name_cannot_label_two_visible_tracks(self):
        t=PersonTracker()
        tracks=t.update([obs(0,0),obs(160,1)],0)
        self.confirm(t,tracks[0],'Alice')
        self.confirm(t,tracks[1],'Alice')
        self.assertIsNone(tracks[1].name)
        self.assertTrue(tracks[1].uncertain)

    def test_face_body_association_one_to_one(self):
        face=lambda x: SimpleNamespace(box=np.array([x,10,x+30,45],np.float32))
        bodies=[np.array([0,0,80,240]),np.array([100,0,180,240])]
        self.assertEqual(associate_faces([face(20),face(120)],bodies),{0:0,1:1})
        # Two identical head regions must not cause the same face to identify both.
        self.assertEqual(associate_faces([face(20)],[bodies[0],bodies[0]]),{})

    def test_faces_below_head_region_are_not_assigned(self):
        face=SimpleNamespace(box=np.array([20,180,50,215]))
        self.assertEqual(associate_faces([face],[np.array([0,0,80,240])]),{})

    def test_unknown_track_can_recover_to_enroll_after_overlap(self):
        t=PersonTracker()
        track=t.update([obs(occluded=True)],0)[0]
        for now in (.2,.4,.6): track=t.update([obs()],now)[0]
        self.assertFalse(track.uncertain)
        self.assertIsNone(track.name)

    def test_named_uncertainty_requires_face_reconfirmation(self):
        t=PersonTracker()
        track=t.update([obs()],0)[0]
        self.confirm(t,track,'Alice')
        t.update([obs(occluded=True)],.6)
        for now in (.8,1.,1.2): track=t.update([obs()],now)[0]
        self.assertTrue(track.uncertain)
        self.assertIsNone(track.display_name(1.2))
        self.confirm(t,track,'Alice',1.4)
        self.assertEqual(track.display_name(1.8),'Alice')

    def test_two_second_turn_gap_uses_motion_and_moderate_appearance(self):
        t=PersonTracker()
        original=t.update([obs()],0)[0]
        t.update([],1)
        changed=obs(5)
        changed.feature=feature(0)*.6+feature(1)*.8
        recovered=t.update([changed],2)[0]
        self.assertEqual(recovered.track_id,original.track_id)

    def test_prediction_does_not_extrapolate_box_shrinking(self):
        t=PersonTracker()
        track=t.update([obs()],0)[0]
        narrow=obs();narrow.box=np.array([20,0,60,240],np.float32)
        track=t.update([narrow],.1)[0]
        predicted=track.prediction(.8)
        self.assertAlmostEqual(predicted[2]-predicted[0],40.)

    def test_old_fragment_does_not_steal_live_track(self):
        t=PersonTracker()
        old=t.update([obs()],0)[0]
        old.visible=False
        live=PersonTrack(2,obs(5).box,.2,.4)
        live.features.append(feature(0))
        t.tracks[2]=live;t.next_id=3
        result=t.update([obs()],.6)
        self.assertEqual(result[0].track_id,2)
        self.assertFalse(result[0].uncertain)
        self.assertFalse(old.visible)


class DatabaseTests(unittest.TestCase):
    def test_template_format_and_runner_up_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'faces.npz'
            np.savez(p,Alice=feature(0),Bob=feature(1))
            db=FaceDatabase(p)
            self.assertEqual(db.match(feature(0))[0],'Alice')
            tie=(feature(0)+feature(1))/np.sqrt(2)
            self.assertIsNone(db.match(tie)[0])
            db.enroll('Charlie',[feature(2)]*8)
            db.reload()
            self.assertEqual(db.match(feature(2))[0],'Charlie')
            self.assertEqual(db.entries['Charlie'].shape,(8,512))

    def test_invalid_enrollment_does_not_overwrite_database(self):
        with tempfile.TemporaryDirectory() as directory:
            db=FaceDatabase(Path(directory)/'faces.npz')
            with self.assertRaises(ValueError): db.enroll('',[feature(0)]*8)
            with self.assertRaises(ValueError): db.enroll('A',[feature(0)]*2)
            self.assertFalse(db.path.exists())


if __name__=='__main__': unittest.main()
