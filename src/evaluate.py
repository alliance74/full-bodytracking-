import argparse
from pathlib import Path
import cv2, numpy as np

from .embed import ArcFaceEmbedderONNX


def describe(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "n=0"
    return (f"n={arr.size} mean={arr.mean():.3f} std={arr.std():.3f} "
            f"p05={np.percentile(arr, 5):.3f} p50={np.percentile(arr, 50):.3f} "
            f"p95={np.percentile(arr, 95):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Threshold evaluation from enrolled aligned crops.")
    parser.add_argument('--model', default='models/embedder_arcface.onnx')
    parser.add_argument('--enroll-dir', default='data/enroll')
    parser.add_argument('--target-far', type=float, default=0.01)
    args = parser.parse_args()

    enroll_dir = Path(args.enroll_dir)
    people = [p for p in enroll_dir.iterdir() if p.is_dir()] if enroll_dir.exists() else []
    if not people:
        print(f"No enrolled people found under {enroll_dir}. Run enrollment first.")
        return

    emb = ArcFaceEmbedderONNX(model_path=args.model)
    per = {}
    for p in people:
        xs = []
        for f in sorted(p.glob('*.jpg')):
            im = cv2.imread(str(f))
            if im is not None and im.shape[:2] == (112, 112):
                xs.append(emb.embed(im))
        if len(xs) >= 2:
            per[p.name] = xs
        else:
            print(f"Skipping {p.name}: only {len(xs)} valid aligned crops (need >= 2).")

    names = sorted(per)
    if not names:
        print("Not enough data to evaluate. Enroll more samples.")
        return

    gen, imp = [], []
    for n in names:
        for i in range(len(per[n])):
            for j in range(i + 1, len(per[n])):
                gen.append(1 - float(np.dot(per[n][i], per[n][j])))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for a in per[names[i]]:
                for b in per[names[j]]:
                    imp.append(1 - float(np.dot(a, b)))

    genuine = np.array(gen, dtype=np.float32)
    impostor = np.array(imp, dtype=np.float32)

    print("\n=== Distance Distributions (cosine distance = 1 - cosine similarity) ===")
    print(f"Genuine (same person):  {describe(genuine)}")
    print(f"Impostor (diff persons): {describe(impostor)}")
    if impostor.size == 0:
        print("\nOnly one identity enrolled -> impostor set is empty. Enroll at least two people for a meaningful FAR.")

    thresholds = np.arange(0.10, 1.20 + 1e-9, 0.01, dtype=np.float32)
    results = []
    for thr in thresholds:
        far = float(np.mean(impostor <= thr)) if impostor.size else 0.0
        frr = float(np.mean(genuine > thr)) if genuine.size else 0.0
        results.append((float(thr), far, frr))

    print("\n=== Threshold Sweep (10 rows sampled) ===")
    stride = max(1, len(results) // 10)
    for thr, far, frr in results[::stride]:
        print(f"thr={thr:.2f} FAR={far * 100:5.2f}% FRR={frr * 100:5.2f}%")

    best = None
    for thr, far, frr in results:
        if far <= args.target_far and (best is None or frr < best[2]):
            best = (thr, far, frr)

    if best is not None:
        thr, far, frr = best
        print(f"\nSuggested threshold (target FAR {args.target_far * 100:.1f}%): "
              f"thr={thr:.2f} FAR={far * 100:.2f}% FRR={frr * 100:.2f}%")
        print(f"(Equivalent cosine similarity threshold ~ {1.0 - thr:.3f})")
    else:
        print(f"\nNo threshold met FAR <= {args.target_far * 100:.1f}%. "
              "Try widening the sweep range or collecting more varied samples.")


if __name__ == '__main__':
    main()
