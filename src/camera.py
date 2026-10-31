import argparse
import time

import cv2

from .camera_utils import add_camera_arg, open_camera


def main():
    parser = argparse.ArgumentParser(description="Camera sanity check (open, read, show FPS).")
    add_camera_arg(parser)
    args = parser.parse_args()

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not opened: {args.camera!r}. Try another index or device path.")
    print(f"Camera test on {args.camera!r}. Press q to quit.")

    t0 = time.time()
    frames = 0
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break
        frames += 1
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Camera Test', frame)
        if cv2.waitKey(1) & 255 == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
