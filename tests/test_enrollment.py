import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from src.enrollment import GuidedEnrollment,STEPS
from src.face_database import FaceDatabase,BODY_VIEWS
from src.person_tracking import PersonTracker,Observation
from test_tracking import feature,obs


def subject(vector=None):
    return SimpleNamespace(track_id=1,visible=True,uncertain=False,
                           face=SimpleNamespace(vector=feature(0) if vector is None else vector))


def observation():
    return Observation(np.array([50,40,250,420],np.float32),.9,feature(1),False)


class GuidedTests(unittest.TestCase):
    def test_ready_required_and_countdown_respected(self):
        g=GuidedEnrollment(1,'Alice')
        t=subject()
        g.update(t,observation(),0,(480,640,3))
        self.assertFalse(g.faces)
        g.ready(0)
        self.assertEqual(g.snapshot(1)['countdown'],2)
        g.update(t,observation(),2.9,(480,640,3))
        self.assertFalse(g.faces)
        g.update(t,observation(),3.,(480,640,3))
        self.assertEqual(len(g.faces),1)

    def test_all_views_capture_and_return_face_required_to_save(self):
        g=GuidedEnrollment(1,'Alice')
        t=subject()
        now=0.
        for key,_,count in STEPS:
            self.assertEqual(g.snapshot(now)['step'],key)
            g.ready(now)
            for i in range(count):
                now+=3.1 if i==0 else .5
                if key=='back': t.face=None
                else: t.face=SimpleNamespace(vector=feature(0))
                g.update(t,observation(),now,(480,640,3))
            self.assertIsNone(g.capture_at)
        self.assertTrue(g.complete)
        self.assertEqual(len(g.faces),8)
        self.assertEqual(len(g.bodies),15)
        self.assertEqual(set(g.views),set(BODY_VIEWS))
        t.face=None
        self.assertFalse(g.save_allowed(t))
        t.face=SimpleNamespace(vector=feature(0))
        self.assertTrue(g.save_allowed(t))
        t.face=SimpleNamespace(vector=feature(4))
        self.assertFalse(g.save_allowed(t))

    def test_losing_selected_person_blocks_save(self):
        g=GuidedEnrollment(1,'Alice')
        g.update(subject(),observation(),0,(480,640,3))
        g.update(None,None,2,(480,640,3))
        self.assertFalse(g.failed)
        self.assertTrue(g.paused)
        self.assertFalse(g.save_allowed(subject()))

    def test_other_people_do_not_interrupt_locked_target(self):
        g=GuidedEnrollment(1,'Alice')
        g.ready(0)
        g.update(subject(),observation(),3,(480,640,3),others_visible=1)
        self.assertEqual(len(g.faces),1)
        self.assertIsNotNone(g.capture_at)

    def test_different_face_pauses_and_keeps_samples(self):
        g=GuidedEnrollment(1,'Alice')
        g.faces=[feature(0)]
        g.update(subject(feature(5)),observation(),0,(480,640,3))
        self.assertFalse(g.failed)
        g.update(subject(feature(5)),observation(),.2,(480,640,3))
        g.update(subject(feature(5)),observation(),.4,(480,640,3))
        self.assertFalse(g.failed)
        self.assertTrue(g.paused)
        self.assertEqual(len(g.faces),1)

    def test_brief_gap_preserves_samples_and_resumes_capture(self):
        g=GuidedEnrollment(1,'Alice')
        g.ready(0)
        g.update(subject(),observation(),3,(480,640,3))
        self.assertEqual(len(g.faces),1)
        g.update(None,None,3.5,(480,640,3))
        for now in (3.8,4.,4.2): g.update(subject(),observation(),now,(480,640,3))
        self.assertFalse(g.paused)
        self.assertFalse(g.failed)
        self.assertEqual(len(g.faces),1)
        g.update(subject(),observation(),5.3,(480,640,3))
        self.assertEqual(len(g.faces),2)

    def test_ambiguity_pauses_without_deleting_completed_samples(self):
        g=GuidedEnrollment(1,'Alice')
        g.faces=[feature(0)]*4
        g.step=1
        t=subject();t.uncertain=True
        g.update(t,observation(),0,(480,640,3))
        self.assertFalse(g.failed)
        self.assertTrue(g.paused)
        self.assertEqual(len(g.faces),4)
        self.assertEqual(g.step,1)

    def test_recover_new_id_requires_three_matching_faces(self):
        g=GuidedEnrollment(1,'Alice')
        g.faces=[feature(0)]*4
        other=subject();other.track_id=2
        for now in (.1,.3):g.recover_track([other],now)
        self.assertEqual(g.track_id,1)
        g.recover_track([other],.5)
        self.assertEqual(g.track_id,2)

    def test_selected_face_anchors_recovery_before_first_sample(self):
        g=GuidedEnrollment(1,'Alice')
        g.anchor(subject())
        self.assertFalse(g.faces)
        other=subject();other.track_id=6
        for now in (.1,.3,.5): g.recover_track([other],now)
        self.assertEqual(g.track_id,6)
        self.assertFalse(g.faces,'Recovery anchors must not count as captured samples')

    def test_bootstrap_never_chooses_an_unrelated_visible_face(self):
        g=GuidedEnrollment(1,'Alice')
        g.anchor(subject())
        other=subject(feature(5));other.track_id=6
        for now in (.1,.3,.5): g.recover_track([other],now)
        self.assertEqual(g.track_id,1)

    def test_recover_refuses_other_face_and_tied_candidates(self):
        g=GuidedEnrollment(1,'Alice')
        g.faces=[feature(0)]*4
        other=subject(feature(5));other.track_id=2
        for now in (.1,.3,.5):g.recover_track([other],now)
        self.assertEqual(g.track_id,1)
        a=subject();a.track_id=2
        b=subject();b.track_id=3
        for now in (.7,.9,1.1):g.recover_track([a,b],now)
        self.assertEqual(g.track_id,1)

    def test_cropped_body_not_captured(self):
        g=GuidedEnrollment(1,'Alice',framing='full')
        g.step=3
        g.ready(0)
        cropped=Observation(np.array([0,0,200,480],np.float32),.9,feature(1))
        g.update(subject(),cropped,3,(480,640,3))
        self.assertFalse(g.bodies)

    def test_upper_body_allows_torso_at_bottom_of_camera_frame(self):
        g=GuidedEnrollment(1,'Alice',framing='upper')
        g.step=3;g.faces=[feature(0)]*8
        g.ready(0)
        torso=Observation(np.array([60,30,400,480],np.float32),.99,feature(1))
        g.update(subject(feature(9)),torso,3,(480,640,3))
        self.assertEqual(len(g.bodies),1)
        self.assertFalse(g.paused)
        self.assertFalse(g.failed)
        self.assertIn('UPPER BODY',g.snapshot(3)['instruction'])

    def test_draft_restores_progress_without_trusting_an_old_track_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'draft.npz'
            g=GuidedEnrollment(7,'Alice')
            g.faces=[feature(0)]*8;g.bodies=[feature(1)]*2
            g.views=['front']*2;g.step=3;g.count=2;g.capture_at=4.
            g.checkpoint(path)
            restored=GuidedEnrollment.restore(path)
            self.assertEqual(restored.track_id,-1)
            self.assertEqual(len(restored.faces),8)
            self.assertEqual(len(restored.bodies),2)
            self.assertEqual(restored.step,3)
            self.assertEqual(restored.count,2)
            self.assertTrue(restored.paused)
            self.assertFalse(restored.save_allowed(subject()))

    def test_upper_body_still_rejects_a_tiny_detection(self):
        g=GuidedEnrollment(1,'Alice')
        g.step=3;g.ready(0)
        tiny=Observation(np.array([60,30,130,100],np.float32),.99,feature(1))
        g.update(subject(),tiny,3,(480,640,3))
        self.assertFalse(g.bodies)
        self.assertIn('closer',g.warning)


class BodyGalleryTests(unittest.TestCase):
    def enroll(self,db,name,vector):
        db.enroll(name,[feature(0)]*8,[vector]*15,[view for view in BODY_VIEWS for _ in range(3)])

    def test_profile_survives_reload_and_face_keys_are_separate(self):
        with tempfile.TemporaryDirectory() as d:
            db=FaceDatabase(Path(d)/'profiles.npz')
            self.enroll(db,'Alice',feature(1))
            other=FaceDatabase(db.path)
            self.assertEqual(list(other.entries),['Alice'])
            self.assertEqual(other.body_entries['Alice'].shape,(15,512))
            self.assertEqual(other.match_body(feature(1))[0],'Alice')
            self.assertEqual(other.match(feature(0))[0],'Alice')

    def test_similar_uniform_profiles_rejected_by_runner_up_gap(self):
        with tempfile.TemporaryDirectory() as d:
            db=FaceDatabase(Path(d)/'profiles.npz')
            self.enroll(db,'Alice',feature(1))
            near=feature(1)*.98+feature(2)*np.sqrt(1-.98**2)
            self.enroll(db,'Bob',near)
            name,score,gap=db.match_body(feature(1))
            self.assertIsNone(name)
            self.assertAlmostEqual(score,1.)
            self.assertLess(gap,.10)

    def test_low_similarity_does_not_assign_name(self):
        with tempfile.TemporaryDirectory() as d:
            db=FaceDatabase(Path(d)/'profiles.npz')
            self.enroll(db,'Alice',feature(1))
            self.assertIsNone(db.match_body(feature(2))[0])

    def test_partial_body_profile_is_not_saved(self):
        with tempfile.TemporaryDirectory() as d:
            db=FaceDatabase(Path(d)/'profiles.npz')
            with self.assertRaises(ValueError):
                db.enroll('Alice',[feature(0)]*8,[feature(1)]*3,['front']*3)
            self.assertFalse(db.path.exists())

    def test_failed_registration_preserves_previous_profile(self):
        with tempfile.TemporaryDirectory() as d:
            db=FaceDatabase(Path(d)/'profiles.npz')
            self.enroll(db,'Alice',feature(1))
            before=db.path.read_bytes()
            with self.assertRaises(ValueError):
                db.enroll('Bob',[feature(0)]*8,[feature(1)]*3,['front']*3)
            self.assertEqual(db.path.read_bytes(),before)

    def test_body_identity_is_unverified_until_face_confirmed(self):
        tracker=PersonTracker()
        t=tracker.update([obs()],0)[0]
        for now in (.2,.4): t=tracker.update([obs()],now)[0]
        tracker.observe_body(t,'Alice',.95,.4)
        tracker.observe_body(t,'Alice',.95,.6)
        self.assertIsNone(t.name)
        tracker.observe_body(t,'Alice',.95,.8)
        self.assertEqual(t.evidence(.8),'body gallery (unverified)')
        self.assertEqual(t.display_name(.8),'Alice')
        for now in (1.,1.2,1.4): tracker.observe_face(t,'Alice',now)
        self.assertEqual(t.identity_source,'face')

    def test_clear_conflicting_face_revokes_body_identity(self):
        tracker=PersonTracker()
        t=tracker.update([obs()],0)[0]
        t.clean_hits=3
        for now in (.2,.4,.6): tracker.observe_body(t,'Alice',.95,now)
        self.assertEqual(t.name,'Alice')
        tracker.observe_face(t,'Bob',.8)
        self.assertIsNone(t.display_name(.8))


if __name__=='__main__': unittest.main()
