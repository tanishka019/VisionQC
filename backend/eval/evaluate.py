"""
Measure VisionQC on a labelled folder, so every change can be judged by numbers.

    python eval/evaluate.py DATA_DIR [--train 25] [--seed 0] [--sheet out.jpg]

DATA_DIR holds:  train_good/  (good parts to learn from),  test_good/  (other good parts),
                 test_bad/    (defective parts; the file name starts with the defect type, e.g. scratch_001.png)

Prints the catch rate on bad parts, the false-reject rate on good parts, AUROC (threshold-free) and the
best threshold, per defect type. Model settings come from the VISIONQC_* environment variables.
"""
import argparse, os, random, sys, tempfile, time
from pathlib import Path

os.environ.setdefault("VISIONQC_DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image
from ml import model

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def files(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in EXT)


def auroc(good, bad):
    g, b = np.array(good), np.array(bad)
    return float(((b[:, None] > g[None]).mean() + 0.5 * (b[:, None] == g[None]).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("--train", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--sheet")
    a = ap.parse_args()
    root = Path(a.data)
    train = files(root / "train_good"); random.Random(a.seed).shuffle(train); train = train[:a.train]
    tmp = Path(tempfile.mkdtemp())
    for p in train:
        (tmp / p.name).symlink_to(p.resolve())
    t0 = time.time(); model.train(tmp, "eval"); t_train = time.time() - t0

    def run(paths):
        t = time.time(); r = [model.inspect(p, 0.5) for p in paths]; return r, (time.time() - t) / max(1, len(paths))
    good, per_img = run(files(root / "test_good")); bad_paths = files(root / "test_bad"); bad, _ = run(bad_paths)
    gs, bs = [r["score"] for r in good], [r["score"] for r in bad]
    kinds = ["_".join(p.stem.split("_")[:2]) if p.stem.startswith(("scratch", "thread", "manipulated")) else p.stem.split("_")[0] for p in bad_paths]

    print(f"train photos: {len(train)}   train {t_train:.0f}s   inspect {per_img:.2f}s/img")
    print(f"AUROC {auroc(gs, bs):.3f}")
    for thr in (0.5, 0.4, 0.3):
        print(f"threshold {thr}: catch {np.mean(np.array(bs) >= thr):.0%}   false reject {np.mean(np.array(gs) >= thr):.0%}")
    cand = np.linspace(0.05, 0.95, 91)
    best = max(cand, key=lambda t: np.mean(np.array(bs) >= t) - np.mean(np.array(gs) >= t))
    print(f"best threshold {best:.2f}: catch {np.mean(np.array(bs) >= best):.0%}   false reject {np.mean(np.array(gs) >= best):.0%}")
    for k in sorted(set(kinds)):
        s = [x for x, kk in zip(bs, kinds) if kk == k]
        print(f"  {k:<22} n={len(s):<3} caught@0.5 {np.mean(np.array(s) >= 0.5):.0%}   median score {np.median(s):.2f}")
    if a.sheet:
        pick = [bad[i] for i in range(0, len(bad), max(1, len(bad) // 8))][:8]
        tiles = [Image.open(r["heatmap_path"]).convert("RGB").resize((300, 300)) for r in pick]
        sheet = Image.new("RGB", (300 * 4, 300 * ((len(tiles) + 3) // 4)))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 4) * 300, (i // 4) * 300))
        sheet.save(a.sheet, quality=85)


if __name__ == "__main__":
    main()
