import argparse

import cv2
import mediapipe as mp

from .camera_utils import add_camera_arg, open_camera

IDX = [33, 263, 1, 61, 291]
NAMES = ["L-eye", "R-eye", "Nose", "L-mouth", "R-mouth"]


def main():
    parser = argparse.ArgumentParser(description="5-point facial landmark sanity check.")
    add_camera_arg(parser)
    args = parser.parse_args()

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not opened: {args.camera!r}")
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
    print(f"5pt landmarks on {args.camera!r}. Press q to quit.")
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        for x, y, bw, bh in cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60)):
            cv2.rectangle(f, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        r = mesh.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if r.multi_face_landmarks:
            lm = r.multi_face_landmarks[0].landmark
            for i, name in zip(IDX, NAMES):
                px, py = int(lm[i].x * w), int(lm[i].y * h)
                cv2.circle(f, (px, py), 4, (0, 255, 0), -1)
                cv2.putText(f, name, (px + 6, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow('5pt Landmarks', f)
        if cv2.waitKey(1) & 255 == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
