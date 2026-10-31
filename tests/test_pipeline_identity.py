import tempfile
import unittest
from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from src.pipeline import PersonPipeline
from src.face_database import FaceDatabase,BODY_VIEWS
from src.person_tracking import PersonTracker
from src.vision_models import Face
from test_tracking import feature,obs


class PipelineIdentityTests(unittest.TestCase):
    def make_pipeline(self,path,face_vector):
        p=PersonPipeline.__new__(PersonPipeline)
        box=np.array([0,0,80,240],np.float32)
        p.people=SimpleNamespace(threshold=.35,detect=lambda f:[(box,.99)])
        p.faces=SimpleNamespace(detect=lambda f:[Face(np.array([20,10,60,70]),.99,np.zeros((5,2),np.float32))])
        p.arcface=SimpleNamespace(embed=lambda f:face_vector)
        p.body=SimpleNamespace(embed_many=lambda crops:[feature(1) for c in crops])
        p.database=FaceDatabase(path)
        p.database.enroll('Alice',[feature(0)]*8,[feature(1)]*15,[v for v in BODY_VIEWS for _ in range(3)])
        p.tracker=PersonTracker()
        track=p.tracker.update([obs(identity=1)],0)[0]
        track.name='Alice';track.identity_source='body gallery';track.body_identity_at=0
        p.commands=SimpleQueue();p.enrolling=None;p.save_requested=False
        p.body_threshold=.90;p.body_margin=.10;p.message=''
        return p

    def test_unknown_visible_face_hides_outfit_only_name(self):
        with tempfile.TemporaryDirectory() as directory:
            p=self.make_pipeline(Path(directory)/'profiles.npz',feature(9))
            with patch('src.pipeline.face_quality',return_value=True),patch('src.pipeline.align_face',return_value=np.zeros((112,112,3),np.uint8)):
                result=p.process(np.zeros((300,300,3),np.uint8),.2)
            self.assertIsNone(result.tracks[0]['name'])
            self.assertEqual(result.tracks[0]['evidence'],'uncertain')

    def test_matching_face_upgrades_outfit_guess_after_three_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            p=self.make_pipeline(Path(directory)/'profiles.npz',feature(0))
            with patch('src.pipeline.face_quality',return_value=True),patch('src.pipeline.align_face',return_value=np.zeros((112,112,3),np.uint8)):
                for now in (.2,.4,.6): result=p.process(np.zeros((300,300,3),np.uint8),now)
            self.assertEqual(result.tracks[0]['name'],'Alice')
            self.assertEqual(result.tracks[0]['evidence'],'face confirmed')
            self.assertEqual(result.tracks[0]['face_similarity'],1.)


if __name__=='__main__':unittest.main()
