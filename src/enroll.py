import argparse
from pathlib import Path
import cv2, numpy as np, time, json

from .camera_utils import add_camera_arg, open_camera
from .haar_5pt import Haar5ptDetector, align_face_5pt
from .embed import ArcFaceEmbedderONNX

DB = Path('data/db')
CROPS = Path('data/enroll')


def main():
    parser = argparse.ArgumentParser(description="Enroll a person's face into the recognition database.")
    add_camera_arg(parser)
    parser.add_argument('--name', default=None, help="Person's name. Prompted interactively if omitted.")
    parser.add_argument('--model', default='models/embedder_arcface.onnx')
    parser.add_argument('--samples-needed', type=int, default=15)
    args = parser.parse_args()

    name = args.name.strip() if args.name else input('Enter person name: ').strip()
    if not name:
        return
    DB.mkdir(parents=True, exist_ok=True)
    person = CROPS / name
    person.mkdir(parents=True, exist_ok=True)
    emb = ArcFaceEmbedderONNX(model_path=args.model)
    det = Haar5ptDetector(debug=False)
    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not opened: {args.camera!r}")
    samples = []
    auto = False
    last = 0
    needed = args.samples_needed
    print('SPACE=capture | a=auto capture | s=save | q=quit')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = det.detect(frame)
        aligned = None
        vis = frame.copy()
        if faces:
            f = faces[0]
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            for x, y in f.kps.astype(int):
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
            aligned, _ = align_face_5pt(frame, f.kps)
            cv2.imshow('aligned', aligned)
        cv2.putText(vis, f'{name} samples={len(samples)}/{needed}  SPACE capture  A auto  S save  Q quit',
                    (10, 30), 0, 0.55, (255, 255, 255), 2)
        cv2.imshow('enroll', vis)
        key = cv2.waitKey(1) & 255
        now = time.time()
        if auto and aligned is not None and now - last > .25:
            samples.append(emb.embed(aligned))
            cv2.imwrite(str(person / f'{int(now * 1000)}.jpg'), aligned)
            last = now
        if key == ord('a'):
            auto = not auto
        elif key == ord(' '):
            if aligned is not None:
                samples.append(emb.embed(aligned))
                cv2.imwrite(str(person / f'{int(now * 1000)}.jpg'), aligned)
                print('captured', len(samples))
        elif key == ord('s'):
            if len(samples) < 5:
                print('Capture at least 5 samples first')
                continue
            v = np.mean(np.stack(samples), axis=0)
            v = v / (np.linalg.norm(v) + 1e-12)
            old = {}
            p = DB / 'face_db.npz'
            if p.exists():
                d = np.load(p, allow_pickle=True)
                old = {k: d[k].astype(np.float32) for k in d.files}
            old[name] = v.astype(np.float32)
            np.savez(p, **old)
            (DB / 'face_db.json').write_text(json.dumps(
                {'names': sorted(old), 'embedding_dim': int(v.size), 'samples': len(samples)}, indent=2))
            print('SAVED', name, 'identities=', len(old))
            break
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
