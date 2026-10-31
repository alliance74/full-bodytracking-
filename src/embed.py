from __future__ import annotations
import argparse
import cv2, numpy as np, onnxruntime as ort


class ArcFaceEmbedderONNX:
    def __init__(self, model_path='models/embedder_arcface.onnx', input_size=(112, 112), debug=False):
        self.w, self.h = input_size
        options = ort.SessionOptions()
        # Small batches do better with a bounded worker pool on classroom CPUs;
        # leaving this to ONNX Runtime can spawn one worker per logical core.
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(model_path, sess_options=options,
                                         providers=['CPUExecutionProvider'])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        if debug:
            print('input', self.sess.get_inputs()[0].shape, 'output', self.sess.get_outputs()[0].shape)

    def _forward(self, img):
        if img.shape[:2] != (self.h, self.w):
            img = cv2.resize(img, (self.w, self.h))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = np.transpose((rgb - 127.5) / 128.0, (2, 0, 1))[None]
        return self.sess.run([self.out_name], {self.in_name: x.astype(np.float32)})[0].reshape(-1).astype(np.float32)

    def embed(self, img):
        raw = self._forward(img)
        return raw / (np.linalg.norm(raw) + 1e-12)

    def embed_with_norm(self, img):
        """Like embed(), but also returns the pre-normalization L2 norm (for diagnostics)."""
        raw = self._forward(img)
        norm = float(np.linalg.norm(raw) + 1e-12)
        return raw / norm, norm


def main():
    from .camera_utils import add_camera_arg, open_camera
    from .haar_5pt import Haar5ptDetector, align_face_5pt

    parser = argparse.ArgumentParser(description="ArcFace ONNX embedding sanity check.")
    add_camera_arg(parser)
    parser.add_argument('--model', default='models/embedder_arcface.onnx')
    args = parser.parse_args()

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not opened: {args.camera!r}")
    det = Haar5ptDetector()
    embedder = ArcFaceEmbedderONNX(model_path=args.model, debug=True)

    prev = None
    print(f"Embedding demo on {args.camera!r}. Press q to quit.")
    while True:
        ok, f = cap.read()
        if not ok:
            break
        vis = f.copy()
        faces = det.detect(f)
        if faces:
            face = faces[0]
            cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
            aligned, _ = align_face_5pt(f, face.kps)
            emb, norm_before = embedder.embed_with_norm(aligned)
            lines = [f"embedding dim: {emb.size}", f"norm(before L2): {norm_before:.2f}"]
            if prev is not None:
                lines.append(f"cos(prev,this): {float(np.dot(prev, emb)):.3f}")
            prev = emb
            y = 30
            for line in lines:
                cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                y += 26
            cv2.imshow('aligned', aligned)
        cv2.imshow('Face Embedding', vis)
        if cv2.waitKey(1) & 255 == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
