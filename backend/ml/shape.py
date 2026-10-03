"""
Shape check for single, elongated parts (screws, bolts, pins, rods).

PatchCore looks at small patches, so a gently bent part has nothing "locally" wrong with it and passes. For
a single long object on a plain background we therefore also measure how straight it is: how far its
centre line wanders from a straight line, relative to its length. This does not depend on rotation or size.
It switches itself off (returns None) when the photos do not show one long object on a plain background.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

WORK_SIDE   = 256     # px, long side used for the measurement
MIN_ELONG   = 2.0     # principal-axis length ratio needed to count as "long"
BINS        = 28      # slices along the part used to trace its centre line
MIN_BINS    = 12
MIN_FRAC, MAX_FRAC = 0.002, 0.45   # foreground share of the photo that is plausible for one part


def _foreground(arr: np.ndarray) -> np.ndarray | None:
    border = np.concatenate([arr[:4].reshape(-1, 3), arr[-4:].reshape(-1, 3),
                             arr[:, :4].reshape(-1, 3), arr[:, -4:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    # plain-background check: the border must be uniform
    if np.percentile(np.abs(border - bg).max(axis=1), 95) > 25:
        return None
    # remove slow lighting gradients, then threshold the colour difference
    diff = np.abs(arr - bg).max(axis=2)
    diff = ndimage.gaussian_filter(diff, 1.0)
    thr = max(18.0, 0.35 * float(np.percentile(diff, 99.5)))
    mask = diff > thr
    mask = ndimage.binary_opening(mask, iterations=1)
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = lab == (1 + int(np.argmax(sizes)))
    frac = keep.mean()
    return keep if MIN_FRAC <= frac <= MAX_FRAC else None


def measure(img: Image.Image) -> dict | None:
    """Return {"dev": float, "mask": bool array, "resid": float array} or None when not applicable.

    dev   = largest distance of the centre line from a straight line, as a fraction of the part's length.
    resid = per-pixel version of that distance (for drawing), same shape as mask.
    """
    img = img.convert("RGB")
    s = WORK_SIDE / max(img.size)
    arr = np.asarray(img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR), dtype=float)
    mask = _foreground(arr)
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(float)
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    w, vec = np.linalg.eigh(cov)
    if w[0] <= 1e-9 or np.sqrt(w[1] / w[0]) < MIN_ELONG:
        return None
    u_ax, v_ax = vec[:, 1], vec[:, 0]
    u, v = (pts - c) @ u_ax, (pts - c) @ v_ax
    lo, hi = np.percentile(u, [1, 99])
    length = hi - lo
    if length < 20:
        return None
    edges = np.linspace(lo, hi, BINS + 1)
    idx = np.clip(np.digitize(u, edges) - 1, 0, BINS - 1)
    cu, cv, ok = [], [], []
    for b in range(BINS):
        sel = idx == b
        if sel.sum() >= 5:
            cu.append(u[sel].mean()); cv.append(np.median(v[sel])); ok.append(b)
    if len(ok) < MIN_BINS:
        return None
    cu, cv = np.array(cu), np.array(cv)
    # straight line through the centre line, fitted robustly (the head / tip can be fat or thin)
    a, b0 = np.polyfit(cu, cv, 1)
    res = np.abs(cv - (a * cu + b0))
    dev = float(res.max() / length)
    per_bin = np.zeros(BINS); per_bin[ok] = res
    resid = np.zeros(mask.shape)
    resid[ys, xs] = per_bin[idx] / length
    return {"dev": dev, "mask": mask, "resid": ndimage.gaussian_filter(resid, 2.0)}


# ── alignment ────────────────────────────────────────────────────────────────
def align(img: Image.Image, canvas: tuple[int, int], fill_frac: float = 0.9):
    """Rotate, scale and crop a single long part so it always lies the same way, filling the frame.

    Without this a screw photographed at a random angle and position gives the model a different "normal" for
    every pose, and a small defect is a few pixels of a big picture. Returns (aligned image, M) where M is the
    2x3 affine matrix original → aligned (so heat can be mapped back), or None when the photo does not show
    one long part on a plain background. The thicker end (a screw's head) is always put on the left.
    """
    import cv2
    img = img.convert("RGB")
    s = WORK_SIDE / max(img.size)
    small = np.asarray(img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR), dtype=float)
    mask = _foreground(small)
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(float) / s          # original pixel coordinates
    c = pts.mean(axis=0)
    w, vec = np.linalg.eigh(np.cov((pts - c).T))
    if w[0] <= 1e-9 or np.sqrt(w[1] / w[0]) < MIN_ELONG:
        return None
    u_ax = vec[:, 1]
    u = (pts - c) @ u_ax
    lo, hi = np.percentile(u, [1, 99])
    length = hi - lo
    if length < 20 / s:
        return None
    v_ax = np.array([-u_ax[1], u_ax[0]])                         # a pure rotation, never a mirror image
    v = (pts - c) @ v_ax
    span = 0.15 * length
    if v[u > hi - span].std() > v[u < lo + span].std():          # thicker end → left
        u_ax, v_ax = -u_ax, -v_ax
        u, v = -u, -v
        lo, hi = -hi, -lo
    vlo, vhi = np.percentile(v, [1, 99])
    W, H = canvas
    k = W * fill_frac / length
    M = np.zeros((2, 3))
    for row, ax, mid, half in ((0, u_ax, (lo + hi) / 2, W / 2), (1, v_ax, (vlo + vhi) / 2, H / 2)):
        M[row, :2] = k * ax
        M[row, 2] = -k * (ax @ c + mid) + half
    bg = tuple(float(x) for x in np.median(np.concatenate([small[:4].reshape(-1, 3), small[-4:].reshape(-1, 3)]), axis=0))
    out = cv2.warpAffine(np.asarray(img), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    return Image.fromarray(out), M


def part_mask(img: Image.Image, margin: float = 0.03) -> np.ndarray | None:
    """Full-size boolean mask of the part plus a margin (None if no single part is found). Used to keep
    background dust and shadows out of the heatmap."""
    img = img.convert("RGB")
    s = WORK_SIDE / max(img.size)
    small = np.asarray(img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR), dtype=float)
    mask = _foreground(small)
    if mask is None:
        return None
    mask = ndimage.binary_dilation(mask, iterations=max(2, int(margin * WORK_SIDE)))
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(img.size, Image.NEAREST)) > 0
