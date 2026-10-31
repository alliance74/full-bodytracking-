import argparse

import cv2

from .camera_utils import add_camera_arg, open_camera
from .haar_5pt import Haar5ptDetector, align_face_5pt


def main():
    parser = argparse.ArgumentParser(description="5-point face alignment sanity check (112x112).")
    add_camera_arg(parser)
    args = parser.parse_args()

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not opened: {args.camera!r}")
    det = Haar5ptDetector()
    print(f"Alignment on {args.camera!r}. Press q to quit.")
    while True:
        ok, f = cap.read()
        if not ok:
            break
        faces = det.detect(f)
        vis = f.copy()
        if faces:
            face = faces[0]
            cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
            for x, y in face.kps.astype(int):
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
            a, _ = align_face_5pt(f, face.kps)
            cv2.imshow('aligned 112x112', a)
        cv2.imshow('camera', vis)
        if cv2.waitKey(1) & 255 == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
