import argparse

import cv2

from .camera_utils import add_camera_arg, open_camera


def main():
    parser = argparse.ArgumentParser(description="Haar face detection sanity check.")
    add_camera_arg(parser)
    args = parser.parse_args()

    c = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if c.empty():
        raise RuntimeError("Failed to load Haar cascade")
    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not opened: {args.camera!r}")
    print(f"Haar face detect on {args.camera!r}. Press q to quit.")
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        for x, y, w, h in c.detectMultiScale(g, 1.1, 5, minSize=(60, 60)):
            cv2.rectangle(f, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Face Detection', f)
        if cv2.waitKey(1) & 255 == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
