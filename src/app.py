"""Software-only face + body identity tracking, continuing the original project."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import time
import cv2
import numpy as np

from .face_database import ROOT
from .pipeline import PersonPipeline

WINDOW = 'Person Tracking Intelligence'
BASE_HEADER, GUIDE_HEADER, FOOTER = 118, 262, 62
BUTTONS = ('Enroll', 'Ready', 'Save profile', 'Cancel', 'Lock', 'Reload', 'Quit')
COLORS = {'face confirmed': (70, 225, 100), 'body continuity': (235, 180, 65),
          'body reacquired': (230, 180, 65), 'uncertain': (50, 180, 255),
          'body gallery (unverified)': (50, 180, 255),
          'unknown': (160, 165, 175)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera', default='1', help='External camera index, laptop index 0, or video path')
    p.add_argument('--video', help='Recorded video input; uses media timestamps for track expiry')
    p.add_argument('--models', type=Path, default=ROOT/'models')
    p.add_argument('--database', type=Path, default=ROOT/'data/db/face_db.npz')
    p.add_argument('--face-threshold', type=float, default=.38, help='Maximum ArcFace cosine distance; smaller is stricter')
    p.add_argument('--person-threshold', type=float, default=.35)
    p.add_argument('--body-threshold',type=float,default=.90,help='Minimum saved-body cosine similarity; not an accuracy percentage')
    p.add_argument('--body-margin',type=float,default=.10,help='Minimum body similarity gap from next enrolled person')
    p.add_argument('--body-framing',choices=('upper','full'),default='upper',help='Upper body is the default for laptop cameras; full requires head-to-feet framing')
    p.add_argument('--max-lost', type=float, default=8., help='Seconds a missing track can be considered for re-association')
    p.add_argument('--face-hold', type=float, default=60., help='Seconds before an unrefreshed name is hidden until face confirmation')
    p.add_argument('--max-people', type=int, default=12)
    p.add_argument('--width', type=int, default=960)
    p.add_argument('--height', type=int, default=540)
    p.add_argument('--no-mirror', action='store_true', help='Disable the existing horizontal mirror; video stays upright')
    p.add_argument('--headless', action='store_true', help='Process input without opening camera windows')
    p.add_argument('--max-frames', type=int, default=0, help='Stop after N processed frames; 0 means unlimited')
    p.add_argument('--output', type=Path, help='Save annotated MP4, using input video frame rate or 10 FPS for camera')
    p.add_argument('--events', type=Path, help='Save JSONL identity/visibility changes; no frames saved by default')
    return p


def button_rectangles(width, height):
    gap, margin = 8, 12
    bw = (width - 2*margin - (len(BUTTONS)-1)*gap)//len(BUTTONS)
    return [(margin+i*(bw+gap), height-48, margin+i*(bw+gap)+bw, height-8)
            for i in range(len(BUTTONS))]


def render(result, selected=None, locked=None, message='', age=0.):
    frame = result.frame
    h, w = frame.shape[:2]
    header, footer = (GUIDE_HEADER if result.enrollment else BASE_HEADER), FOOTER
    canvas = np.full((h+header+footer, w, 3), (23, 27, 34), np.uint8)
    canvas[header:header+h] = frame
    for face in result.faces:
        x1, y1, x2, y2 = face.box.astype(int)
        cv2.rectangle(canvas, (x1, y1+header), (x2, y2+header), (130, 170, 130), 1)
    for track in result.tracks:
        if not track['visible']:
            continue
        x1, y1, x2, y2 = track['box']
        x1, x2 = int(np.clip(x1,0,w-1)), int(np.clip(x2,0,w-1))
        y1, y2 = int(np.clip(y1,0,h-1))+header, int(np.clip(y2,0,h-1))+header
        color = COLORS.get(track['evidence'], (120, 120, 120))
        thickness = 4 if track['track_id'] in (selected, locked) else 2
        cv2.rectangle(canvas, (x1,y1), (x2,y2), color, thickness)
        label = f"P{track['track_id']} | {track['name'] or 'Unknown'}"
        if track['track_id'] == locked:
            label = 'LOCKED ' + label
        cv2.rectangle(canvas,(x1,y1),(min(w-1,x1+360),min(canvas.shape[0]-footer,y1+62)), (23,27,34), -1)
        cv2.putText(canvas,label,(x1+3,y1+17),cv2.FONT_HERSHEY_SIMPLEX,.52,color,1,cv2.LINE_AA)
        cv2.putText(canvas,track['evidence'],(x1+3,y1+35),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
        face_score,body_score=track.get('face_similarity'),track.get('body_similarity')
        score_text=(f'Face match {face_score*100:.0f}%' if face_score is not None else
                    f'Body match {body_score*100:.0f}%' if body_score is not None else
                    'No current identity match score')
        cv2.putText(canvas,score_text,(x1+3,y1+53),cv2.FONT_HERSHEY_SIMPLEX,.44,color,1,cv2.LINE_AA)
        trail = np.asarray([(x,y+header) for x,y in track['trail']],np.int32)
        if len(trail)>1:
            cv2.polylines(canvas,[trail],False,color,1,cv2.LINE_AA)
    visible = sum(t['visible'] for t in result.tracks)
    cv2.putText(canvas, f"FACE + BODY TRACKING | {visible} people | {result.identities} enrolled | {1000/max(result.elapsed_ms,1):.1f} inference FPS",
                (12,23),cv2.FONT_HERSHEY_SIMPLEX,.53,(230,235,245),1,cv2.LINE_AA)
    text = message or result.message
    cv2.putText(canvas,text[:115],(12,47),cv2.FONT_HERSHEY_SIMPLEX,.43,(190,205,225),1,cv2.LINE_AA)
    status = 'Green: face confirmed | Blue: body tracking, face unverified | Orange: uncertain'
    if locked is not None:
        target = next((t for t in result.tracks if t['track_id']==locked),None)
        if target is None or not target['visible']:
            status = f'LOCKED P{locked}: OUT OF VIEW - current position unknown'
        else:
            status = f"LOCKED P{locked}: {target['name'] or 'identity uncertain'} | {target['evidence']}"
    if age > 1.5:
        status = f'PROCESSING DELAY: image is {age:.1f}s old'
    cv2.putText(canvas,status[:115],(12,76),cv2.FONT_HERSHEY_SIMPLEX,.44,(80,190,255),1,cv2.LINE_AA)
    limits=result.thresholds or {"face":.62,"face_margin":.06,"body":.90,"body_margin":.10}
    limit_text=(f"MATCH SIMILARITY: Face >= {limits['face']*100:.0f}% (gap {limits['face_margin']*100:.0f} pts) | "
                f"Body >= {limits['body']*100:.0f}% (gap {limits['body_margin']*100:.0f} pts) | certainty uncalibrated")
    cv2.putText(canvas,limit_text,(12,101),cv2.FONT_HERSHEY_SIMPLEX,.40,(175,190,210),1,cv2.LINE_AA)
    if result.enrollment:
        guide=result.enrollment
        cv2.rectangle(canvas,(8,113),(w-9,header-9),(38,47,61),-1)
        cv2.putText(canvas,f"ENROLLING: {guide['name']}",(18,139),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
        # Wrap long instructions rather than cutting off the required pose.
        import textwrap
        for i,line in enumerate(textwrap.wrap(guide['instruction'],max(30,int(w/10)))):
            cv2.putText(canvas,line,(18,167+i*23),cv2.FONT_HERSHEY_SIMPLEX,.56,(100,225,255),1,cv2.LINE_AA)
        cv2.putText(canvas,guide['detail'][:max(30,int(w/8))],(18,221),cv2.FONT_HERSHEY_SIMPLEX,.46,(210,220,235),1,cv2.LINE_AA)
        cv2.rectangle(canvas,(18,236),(w-20,245),(65,70,80),-1)
        cv2.rectangle(canvas,(18,236),(18+int((w-38)*guide['progress']),245),(90,215,130),-1)
    for title,(x1,y1,x2,y2) in zip(BUTTONS,button_rectangles(w,canvas.shape[0])):
        if title=='Lock' and locked is not None:
            title='Locked' if result.enrollment else 'Unlock'
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(62,88,116),-1)
        cv2.putText(canvas,title,(x1+8,y1+26),cv2.FONT_HERSHEY_SIMPLEX,.53,(245,245,245),1,cv2.LINE_AA)
    return canvas


def main(argv=None):
    p=parser()
    args=p.parse_args(argv)
    if (not 0 < args.face_threshold < 1 or not 0 < args.person_threshold < 1
            or not 0<args.body_threshold<=1 or not 0<args.body_margin<1):
        p.error('Face and person thresholds must be between 0 and 1')
    if args.max_lost <= 0 or args.face_hold <= 0 or args.max_people < 1:
        p.error('Track timeouts and maximum people must be positive')
    cv2.setNumThreads(2)
    try:
        pipeline=PersonPipeline(args.models,args.database,args.face_threshold,args.max_lost,
                                args.face_hold,args.person_threshold,args.max_people,
                                args.body_threshold,args.body_margin,args.body_framing)
    except (FileNotFoundError,ValueError,RuntimeError,cv2.error) as error:
        p.error(str(error))
    source=args.video or args.camera
    source=int(source) if str(source).isdigit() else source
    cap=cv2.VideoCapture(source)
    if not cap.isOpened() and not args.video and isinstance(source,int) and source!=0:
        print(f'Camera {source} unavailable; trying laptop camera 0.',flush=True)
        cap.release()
        cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        p.error(f'Cannot open input {source!r}. Check camera permission, USB connection, or video path.')
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    fps=fps if 1 <= fps <= 120 else 25.
    # File inputs process every frame; camera inputs keep only one pending task
    # and skip excess capture frames so no inference backlog accumulates.
    state={'result':None,'selected':None,'locked':None,'action':None,'message':'','quit':False}
    root=None
    if not args.headless:
        import tkinter as tk
        from tkinter import simpledialog
        root=tk.Tk()
        root.withdraw()
        cv2.namedWindow(WINDOW,cv2.WINDOW_AUTOSIZE)
        def click(event,x,y,flags,param):
            if event!=cv2.EVENT_LBUTTONDOWN or state['result'] is None:
                return
            result=state['result']
            h,w=result.frame.shape[:2]
            header=GUIDE_HEADER if result.enrollment else BASE_HEADER
            for title,(x1,y1,x2,y2) in zip(BUTTONS,button_rectangles(w,h+header+FOOTER)):
                if x1<=x<=x2 and y1<=y<=y2:
                    state['action']=title
                    return
            y-=header
            if result.enrollment:
                state['message']='Enrollment is locked; other people cannot change the target'
                return
            candidates=[t for t in result.tracks if t['visible'] and
                        t['box'][0]<=x<=t['box'][2] and t['box'][1]<=y<=t['box'][3]]
            if candidates:
                candidates.sort(key=lambda t:(t['box'][2]-t['box'][0])*(t['box'][3]-t['box'][1]))
                state['selected']=candidates[0]['track_id']
                state['message']=f"Selected P{state['selected']}"
        cv2.setMouseCallback(WINDOW,click)
    writer=None
    event_file=None
    executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='person-inference')
    pending=None
    processed=0
    captured=0
    previous={}
    capture_finished=False
    def accept(result):
        nonlocal writer,processed,previous
        state['result']=result
        if result.enrollment:
            state['selected']=result.enrollment['track_id']
            state['locked']=result.enrollment['track_id']
        processed+=1
        visible=[t for t in result.tracks if t['visible']]
        if state['selected'] is None and len(visible)==1:
            state['selected']=visible[0]['track_id']
        if event_file:
            current={t['track_id']:(t['name'],t['evidence'],t['visible']) for t in result.tracks}
            for t in result.tracks:
                if previous.get(t['track_id'])!=current[t['track_id']]:
                    event_file.write(json.dumps({'time':result.captured_at,**{k:v for k,v in t.items() if k!='trail'}})+'\n')
            for tid in previous.keys()-current.keys():
                event_file.write(json.dumps({'time':result.captured_at,'track_id':tid,'evidence':'expired'})+'\n')
            event_file.flush()
            previous=current
        if args.output:
            # Export has a fixed canvas even while the interactive guide expands.
            from dataclasses import replace
            image=render(replace(result,enrollment=None),state['selected'],state['locked'])
            if writer is None:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                writer=cv2.VideoWriter(str(args.output),cv2.VideoWriter_fourcc(*'mp4v'),fps if args.video else 10.,(image.shape[1],image.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot create output video: {args.output}')
            writer.write(image)
    try:
        if args.events:
            args.events.parent.mkdir(parents=True,exist_ok=True)
            event_file=args.events.open('w')
        while not state['quit']:
            if pending is not None and pending.done():
                accept(pending.result())
                pending=None
            if args.max_frames and processed>=args.max_frames:
                break
            if not args.headless:
                key=cv2.waitKey(1)&255
                if key in (27,ord('q')):
                    break
                if key==ord('l'): state['action']='Lock'
                if key==32: state['action']='Ready'
                action=state.pop('action',None)
                if action=='Quit': break
                if action=='Lock':
                    if state['result'] is not None and state['result'].enrollment:
                        state['message']='Enrollment target stays locked. Cancel enrollment before changing the lock.'
                    elif state['locked'] is not None:
                        state['locked']=None
                        state['message']='Unlocked'
                    elif state['selected'] is not None:
                        state['locked']=state['selected']
                        state['message']=f"Locked person track P{state['locked']}"
                    else: state['message']='Click a person before locking'
                elif action=='Enroll':
                    if state['selected'] is None:
                        state['message']='Click a person before enrolling'
                    else:
                        name=simpledialog.askstring('Enroll face and body','Name for the selected person:',parent=root)
                        if name and name.strip():
                            state['locked']=state['selected']
                            pipeline.commands.put(('enroll',(state['selected'],name.strip())))
                            state['message']=''
                elif action in ('Ready','Save profile','Cancel','Reload'):
                    pipeline.commands.put(({'Ready':'ready','Save profile':'save','Cancel':'cancel','Reload':'reload'}[action],None))
                    state['message']=''
                if state['result'] is not None:
                    result=state['result']
                    age=0 if args.video else time.monotonic()-result.captured_at
                    cv2.imshow(WINDOW,render(result,state['selected'],state['locked'],state['message'],age))
                    if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:
                        break
            if args.video and pending is not None:
                if args.headless:
                    accept(pending.result())
                    pending=None
                else:
                    time.sleep(.003)
                continue
            if capture_finished:
                if pending is None: break
                time.sleep(.003)
                continue
            ok,frame=cap.read()
            if not ok:
                capture_finished=True
                continue
            captured+=1
            if pending is not None:
                continue
            # Keep the accepted horizontal mirror from the working project;
            # this never flips the image vertically.
            if not args.no_mirror:
                frame=cv2.flip(frame,1)
            if frame.shape[1]>args.width:
                frame=cv2.resize(frame,(args.width,max(1,int(frame.shape[0]*args.width/frame.shape[1]))))
            timestamp=(captured-1)/fps if args.video else time.monotonic()
            pending=executor.submit(pipeline.process,frame,timestamp)
            if args.headless and not args.video:
                accept(pending.result())
                pending=None
    finally:
        executor.shutdown(wait=True,cancel_futures=True)
        cap.release()
        if writer is not None: writer.release()
        if event_file is not None: event_file.close()
        if root is not None:
            cv2.destroyAllWindows()
            root.destroy()
    print(f'Processed {processed} frames. Software-only session complete.',flush=True)


if __name__=='__main__':
    main()
