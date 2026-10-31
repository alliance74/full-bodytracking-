"""CPU inference for YOLOX-Nano, YuNet, ArcFace, and OSNet.

YOLOX preprocessing/decoding follows its official ONNX runtime export.
The ArcFace embedding implementation and alignment template are retained
from the earlier face-locking project. No PyTorch download is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

from .embed import ArcFaceEmbedderONNX


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def nms(boxes, scores, threshold=.45):
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if not len(rest):
            break
        upper = np.maximum(boxes[i, :2], boxes[rest, :2])
        lower = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        intersection = np.maximum(lower - upper, 0).prod(axis=1)
        area_i = np.maximum(boxes[i, 2:] - boxes[i, :2], 0).prod()
        area_r = np.maximum(boxes[rest, 2:] - boxes[rest, :2], 0).prod(axis=1)
        overlap = intersection / np.maximum(area_i + area_r - intersection, 1e-8)
        order = rest[overlap <= threshold]
    return keep


def appearance_box(box, framing="upper"):
    """Use upper-body appearance consistently for enrollment and matching.

    Tall head-to-feet detections use their upper portion; camera-truncated
    torso detections already contain the desired upper-body view.
    """
    result=np.asarray(box,np.float32).copy()
    width=max(float(result[2]-result[0]),1.)
    height=float(result[3]-result[1])
    if framing=='upper' and height/width>=1.8:
        result[3]=result[1]+height*.62
    return result


class PersonDetector:
    def __init__(self, path, threshold=.35, low_threshold=.15, max_people=12):
        self.net = session(path)
        shape = self.net.get_inputs()[0].shape
        self.height, self.width = int(shape[2]), int(shape[3])
        self.input = self.net.get_inputs()[0].name
        self.threshold, self.low_threshold = threshold, low_threshold
        self.max_people = max_people
        grids, steps = [], []
        for stride in (8, 16, 32):
            y, x = np.mgrid[:self.height // stride, :self.width // stride]
            grid = np.stack((x, y), axis=-1).reshape(-1, 2)
            grids.append(grid)
            steps.append(np.full((len(grid), 1), stride))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(steps).astype(np.float32)

    def detect(self, image):
        height, width = image.shape[:2]
        ratio = min(self.height / height, self.width / width)
        h, w = max(1, int(height * ratio)), max(1, int(width * ratio))
        padded = np.full((self.height, self.width, 3), 114, np.uint8)
        padded[:h, :w] = cv2.resize(image, (w, h))
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
        pred = self.net.run(None, {self.input: blob})[0][0].copy()
        if len(pred) != len(self.grid) or pred.shape[1] != 85:
            raise ValueError("Expected official YOLOX raw COCO output (N,85)")
        # COCO class zero is person. Reject boxes classified as another object.
        scores = pred[:, 4] * pred[:, 5]
        mask = (scores >= self.low_threshold) & (pred[:, 5:].argmax(axis=1) == 0)
        raw, grid, steps = pred[mask], self.grid[mask], self.strides[mask]
        scores = scores[mask]
        centers = (raw[:, :2] + grid) * steps
        sizes = np.exp(np.clip(raw[:, 2:4], -10, 10)) * steps
        boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1) / ratio
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, width)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, height)
        keep = nms(boxes, scores)[:self.max_people]
        return [(boxes[i], float(scores[i])) for i in keep
                if boxes[i, 2] - boxes[i, 0] >= 20 and boxes[i, 3] - boxes[i, 1] >= 45]


@dataclass
class Face:
    box: np.ndarray
    score: float
    keypoints: np.ndarray
    vector: np.ndarray | None = None
    name: str | None = None
    distance: float | None = None
    quality: bool = False


class FaceDetector:
    def __init__(self, path, threshold=.8):
        self.detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), threshold, .3, 5000)

    def detect(self, frame):
        height, width = frame.shape[:2]
        scale = min(1., 960. / width)
        small = cv2.resize(frame, (int(width * scale), int(height * scale))) if scale < 1 else frame
        sy, sx = small.shape[0] / height, small.shape[1] / width
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, rows = self.detector.detect(small)
        result = []
        for row in ([] if rows is None else rows):
            x, y, w, h = row[:4] / np.array([sx, sy, sx, sy])
            points = row[4:14].reshape(5, 2) / np.array([sx, sy])
            # YuNet reports right eye, left eye, nose, right mouth, left mouth;
            # alignment expects image-left eye and mouth first.
            if points[0, 0] > points[1, 0]:
                points[[0, 1]] = points[[1, 0]]
            if points[3, 0] > points[4, 0]:
                points[[3, 4]] = points[[4, 3]]
            box = np.array([max(0, x), max(0, y), min(width, x+w), min(height, y+h)], np.float32)
            result.append(Face(box, float(row[14]), points.astype(np.float32)))
        return result


def align_face(frame, points):
    # Same five-point ArcFace template used by the previous project.
    template = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
    transform, _ = cv2.estimateAffinePartial2D(points, template, method=cv2.LMEDS)
    if transform is None:
        return None
    return cv2.warpAffine(frame, transform, (112, 112), borderMode=cv2.BORDER_CONSTANT)


def face_quality(frame, face):
    x1, y1, x2, y2 = face.box.astype(int)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < 40 or face.score < .85:
        return False
    p = face.keypoints
    eye_span = float(np.linalg.norm(p[1] - p[0]))
    if eye_span < 12 or abs(float(p[0, 1] - p[1, 1])) > eye_span * .5:
        return False
    midpoint = (p[0] + p[1]) / 2
    if abs(float(p[2, 0] - midpoint[0])) > eye_span * .65:
        return False
    if not (max(p[0, 1], p[1, 1]) < p[2, 1] < min(p[3, 1], p[4, 1])):
        return False
    sharpness = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    return sharpness >= 25


class BodyEmbedder:
    """OSNet x0.25 MSMT17 ONNX: RGB, ImageNet normalization, 256x128."""
    def __init__(self, path):
        self.net = session(path)
        self.input = self.net.get_inputs()[0].name
        self.shape = self.net.get_inputs()[0].shape

    def embed(self, crop):
        return self.embed_many([crop])[0]

    def embed_many(self, crops):
        if not crops:
            return []
        h = self.shape[2] if isinstance(self.shape[2], int) else 256
        w = self.shape[3] if isinstance(self.shape[3], int) else 128
        # The small published export uses a static batch of 16. Batch people
        # together and pad unused slots, instead of running it once per person.
        batch_size = self.shape[0] if isinstance(self.shape[0], int) else len(crops)
        result = []
        for start in range(0, len(crops), batch_size):
            items = crops[start:start+batch_size]
            blob = np.zeros((batch_size, 3, h, w), np.float32)
            valid = []
            for index, crop in enumerate(items):
                valid.append(crop.size > 0)
                if not crop.size:
                    continue
                rgb = cv2.cvtColor(cv2.resize(crop, (w, h)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
                rgb = (rgb - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
                blob[index] = rgb.transpose(2, 0, 1)
            output = self.net.run(None, {self.input: np.ascontiguousarray(blob)})[0]
            for index, good in enumerate(valid):
                vector = output[index].reshape(-1).astype(np.float32)
                norm = np.linalg.norm(vector)
                result.append(vector / norm if good and norm > 1e-8 and np.isfinite(vector).all() else None)
        return result


MODEL_FILES = {
    "person": "yolox_nano.onnx",
    "face": "face_detection_yunet_2023mar.onnx",
    "body": "osnet_x0_25_msmt17.onnx",
    "arcface": "embedder_arcface.onnx",
}


def model_paths(directory):
    paths = {key: Path(directory) / value for key, value in MODEL_FILES.items()}
    missing = [p.name for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing models: " + ", ".join(missing) + ". Run python -m src.download_models")
    return paths
