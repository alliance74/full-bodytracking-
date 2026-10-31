from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2, numpy as np
import mediapipe as mp

@dataclass
class FaceKpsBox:
    x1:int; y1:int; x2:int; y2:int; score:float; kps:np.ndarray
    landmarks:np.ndarray|None=None

IDX=(33,263,1,61,291)

def _kps_ok(k,min_eye=12.0):
    le,re,no,lm,rm=k; return float(np.linalg.norm(re-le))>=min_eye and lm[1]>no[1] and rm[1]>no[1]

def _bbox(k,padx=.55,padt=.85,padb=1.15):
    xmin,xmax=k[:,0].min(),k[:,0].max(); ymin,ymax=k[:,1].min(),k[:,1].max()
    w=max(1.,xmax-xmin); h=max(1.,ymax-ymin)
    return np.array([xmin-padx*w,ymin-padt*h,xmax+padx*w,ymax+padb*h],np.float32)

def _clip(b,W,H): return np.array([np.clip(b[0],0,W-1),np.clip(b[1],0,H-1),np.clip(b[2],0,W-1),np.clip(b[3],0,H-1)],np.float32)

def _ema(prev,cur,a): return cur.astype(np.float32) if prev is None else (a*prev+(1-a)*cur).astype(np.float32)

def _matrix(k,out=(112,112)):
    dst=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]],np.float32)
    if out!=(112,112): dst*=np.array([out[0]/112,out[1]/112],np.float32)
    M,_=cv2.estimateAffinePartial2D(k.astype(np.float32),dst,method=cv2.LMEDS)
    if M is None: raise RuntimeError('Could not estimate 5-point alignment transform')
    return M.astype(np.float32)

def align_face_5pt(frame,kps,out_size=(112,112)):
    M=_matrix(kps,out_size)
    return cv2.warpAffine(frame,M,(int(out_size[0]),int(out_size[1])),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT),M

class Haar5ptDetector:
    def __init__(self,min_size=(40,40),smooth_alpha=.80,debug=False):
        self.debug=debug; self.min_size=min_size; self.alpha=smooth_alpha
        self.cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        if self.cascade.empty(): raise RuntimeError('Failed to load Haar cascade')
        # We call FaceMesh on separate face ROIs, not consecutive full frames.
        # Tracking mode can carry landmarks from one crop into the next crop,
        # which produces the large displaced boxes seen in multi-person scenes.
        self.mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=.5)
        self.face_detection=mp.solutions.face_detection.FaceDetection(model_selection=1,min_detection_confidence=.45)
        self.prev_box=None; self.prev_kps=None
    def detect(self,frame,max_faces=1)->List[FaceKpsBox]:
        H,W=frame.shape[:2]
        # Keep detector cost bounded on 720p/1080p webcams.
        detect_scale=min(1.0,640.0/max(1,W))
        detect_w=max(1,int(round(W*detect_scale)))
        detect_h=max(1,int(round(H*detect_scale)))
        detect_frame=cv2.resize(frame,(detect_w,detect_h),interpolation=cv2.INTER_AREA) if detect_scale<1 else frame
        rgb=cv2.cvtColor(detect_frame,cv2.COLOR_BGR2RGB)
        # MediaPipe's full-range detector finds smaller/farther faces than the
        # original Haar-only path. Keep Haar as a fallback for profiles/angles.
        detected=self.face_detection.process(rgb)
        candidates=[]
        if detected.detections:
            for item in detected.detections:
                rel=item.location_data.relative_bounding_box
                x1=max(0,int(rel.xmin*detect_w/detect_scale)); y1=max(0,int(rel.ymin*detect_h/detect_scale))
                x2=min(W,int((rel.xmin+rel.width)*detect_w/detect_scale)); y2=min(H,int((rel.ymin+rel.height)*detect_h/detect_scale))
                if x2>x1 and y2>y1:
                    candidates.append((x1,y1,x2-x1,y2-y1,float(item.score[0])))
        if not candidates:
            gray=cv2.cvtColor(detect_frame,cv2.COLOR_BGR2GRAY)
            gray=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
            # Search faces smaller than before and step scale more finely.
            min_size=tuple(max(24,int(round(v*detect_scale))) for v in self.min_size)
            faces=self.cascade.detectMultiScale(gray,1.05,4,minSize=min_size)
            candidates=[(int(x/detect_scale),int(y/detect_scale),int(w/detect_scale),int(h/detect_scale),.5) for x,y,w,h in faces]
        if not candidates:return []
        candidates=sorted(candidates,key=lambda r:r[2]*r[3],reverse=True)[:max_faces]
        out=[]
        for x,y,w,h,score in candidates:
            # Give FaceMesh enough surrounding context, then upscale small ROIs
            # so landmark extraction still works for faces farther from camera.
            mx,my=.25*w,.35*h; rx1=max(0,int(x-mx)); ry1=max(0,int(y-my)); rx2=min(W,int(x+w+mx)); ry2=min(H,int(y+h+my))
            roi=frame[ry1:ry2,rx1:rx2]
            shortest=max(1,min(roi.shape[:2])); longest=max(1,max(roi.shape[:2]))
            factor=min(1.0,320.0/longest)
            if shortest*factor<128:
                factor=max(factor,128.0/shortest)
            if abs(factor-1.0)>1e-3:
                mesh_w=max(1,int(round(roi.shape[1]*factor)))
                mesh_h=max(1,int(round(roi.shape[0]*factor)))
                interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_CUBIC
                mesh_roi=cv2.resize(roi,(mesh_w,mesh_h),interpolation=interpolation)
            else:
                mesh_roi=roi
            res=self.mesh.process(cv2.cvtColor(mesh_roi,cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks: continue
            lm=res.multi_face_landmarks[0].landmark
            # FaceMesh coordinates are normalized to the actual inference ROI.
            # Map them back using the original crop dimensions. Dividing by the
            # requested scale is incorrect when resize rounding changes a pixel,
            # and previously shifted points badly on large (downscaled) ROIs.
            all_points=np.array([[p.x*roi.shape[1]+rx1,p.y*roi.shape[0]+ry1] for p in lm],np.float32)
            k=all_points[list(IDX)].copy()
            if k[0,0]>k[1,0]: k[[0,1]]=k[[1,0]]
            if k[3,0]>k[4,0]: k[[3,4]]=k[[4,3]]
            if not np.isfinite(k).all() or not _kps_ok(k,max(6.,.13*w)): continue
            # Ensure the landmarks belong to this face proposal. Expanded ROI
            # crops can contain a neighboring face, especially in a classroom.
            margin_x,margin_y=.25*w,.30*h
            inside=((k[:,0]>=x-margin_x)&(k[:,0]<=x+w+margin_x)&
                    (k[:,1]>=y-margin_y)&(k[:,1]<=y+h+margin_y))
            if float(inside.mean())<.8: continue
            # Use the detector's own box for display/tracking. Landmarks drive
            # alignment only, so a stray landmark cannot stretch the box across
            # the image or make it appear off-frame.
            b=_clip(np.array([x-.12*w,y-.18*h,x+1.12*w,y+1.22*h],np.float32),W,H)
            out.append(FaceKpsBox(*map(lambda z:int(round(z)),b),score,k.astype(np.float32),all_points))
        return out

    def close(self):
        self.mesh.close()
        self.face_detection.close()
