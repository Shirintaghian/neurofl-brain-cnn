"""Read a raw BIDS tree directly — no preprocessing step, no derived dataset.

The volume conversion (crop, resize, z-score) happens in the DataLoader worker
at __getitem__ time, so nothing is written back to the data directory. That is
the difference between this and prepare_raw.py: same maths, done lazily.

Cost of that choice: each scan is decompressed and resampled on every epoch.
For the cohorts here (tens of scans) that is a few seconds per epoch and buys
you a model that runs against the raw dataset as it sits. Caching in RAM is on
by default since these datasets are small; set cache=False for large ones.
"""
from __future__ import annotations
import csv, pathlib
import numpy as np
import torch
import nibabel as nib


def scan_bids(root) -> list[tuple[str, pathlib.Path]]:
    """(subject, T1w path) for every readable anatomical scan.

    Handles both flat (sub-XX/anat) and session (sub-XX/ses-YY/anat) layouts.
    Dangling git-annex symlinks are skipped: they glob fine and fail on open,
    which would otherwise crash mid-epoch rather than at startup.
    """
    root = pathlib.Path(root)
    out = []
    for pat in ("sub-*/anat/*T1w.nii.gz", "sub-*/ses-*/anat/*T1w.nii.gz"):
        for f in sorted(root.glob(pat)):
            if not f.is_file():
                continue
            sub = next(p for p in f.parts if p.startswith("sub-"))
            out.append((sub, f))
    return out


def read_participants(root) -> dict:
    root = pathlib.Path(root)
    meta = {}
    tsv = root / "participants.tsv"
    if not tsv.is_file():
        return meta
    with tsv.open(encoding="utf-8-sig") as f:      # some BIDS files carry a BOM
        for r in csv.DictReader(f, delimiter="\t"):
            pid = (r.get("participant_id") or "").strip()
            if pid:
                meta[pid] = {
                    "age": (r.get("age") or "").strip(),
                    # `sex` in most datasets, `gender` in others (e.g. ds000102)
                    "sex": (r.get("sex") or r.get("gender") or "").strip().upper(),
                    "group": (r.get("group") or "").strip(),
                }
    return meta


def to_volume(path, out_shape=(64, 64, 64), fast=False) -> np.ndarray:
    """One NIfTI -> normalised float32 volume.

    fast=False (default): full-resolution trilinear resample — best fidelity.
    fast=True: a coarse strided pre-downsample before the resize, several times
    faster at negligible quality cost for these target shapes, and lighter on
    memory-constrained nodes.
    """
    # Load as float32 from the start. Sandbox RAM is tight, so every step below
    # is written to avoid float64 intermediates and extra full-size copies — a
    # single (176,256,256) float64 temporary is ~88 MiB and overflows the jail.
    #
    # nibabel's ArrayProxy applies its scale/offset in float64 and returns a
    # float64 array; np.asanyarray(..., dtype=float32) then copies, so the peak
    # is a float64 whole-volume temporary. Downsample the volume with nibabel's
    # in-image resampling FIRST (reads at reduced resolution) is not available
    # without extra deps, so instead read the un-scaled native array via
    # get_unscaled() (usually int16, ~half the RAM) and apply the scale factor
    # ourselves in float32.
    # Sandbox RAM is very tight (a single full-res float32 volume, ~44 MiB, can
    # fail to allocate), so DOWNSAMPLE FIRST and do every heavy op (clip, crop,
    # z-score) on the small out_shape volume. Peak memory is then one native
    # int16 array (~22 MiB) briefly, never a full-res float32/float64 temporary.
    img = nib.load(str(path))
    try:
        raw = np.asarray(img.dataobj.get_unscaled())   # native dtype, e.g. int16
        sl = img.dataobj.slope
        inter = img.dataobj.inter
    except Exception:
        raw = np.asanyarray(img.dataobj)               # last resort
        sl, inter = None, None

    # Optional coarse strided pre-downsample before the (expensive) trilinear
    # resize: take every other voxel per axis so interpolate runs on ~1/8 the
    # data. Only when fast=True; full resolution is the default.
    if fast:
        step = 2 if min(raw.shape) >= 2 * min(out_shape) else 1
        if step > 1:
            raw = raw[::step, ::step, ::step]

    # Resize the (already-coarse) array to out_shape. torch converts to float32
    # inside interpolate; the result is tiny (64^3).
    t = torch.from_numpy(np.ascontiguousarray(raw)).to(torch.float32)[None, None]
    del raw
    t = torch.nn.functional.interpolate(
        t, size=tuple(out_shape), mode="trilinear", align_corners=False)
    vol = t[0, 0].numpy().copy()                        # small: out_shape
    del t

    # Apply the scale/offset now, on the small volume.
    if sl not in (None, 1.0):
        vol *= np.float32(sl)
    if inter not in (None, 0.0):
        vol += np.float32(inter)

    # Clip bright outliers, then scale to [0, 1] over the (small) volume.
    hi = float(np.percentile(vol, 99.5))
    if hi > 0:
        np.clip(vol, 0.0, hi, out=vol)
        vol /= hi

    brain = vol[vol > 0.01]                              # per-scan z-score
    if brain.size:
        vol = (vol - brain.mean()) / (brain.std() + 1e-6)
    return np.ascontiguousarray(vol, dtype=np.float32)


class RawBIDSDataset(torch.utils.data.Dataset):
    """Volumes straight from a BIDS tree.

    target: "age" (regression) or "sex" (binary classification).
    """

    def __init__(self, root, target="age", shape=(64, 64, 64),
                 indices=None, train=False, cache=True, fast=False):
        self.root = pathlib.Path(root)
        self.shape, self.target, self.train = tuple(shape), target, train
        self.fast = fast
        self.cache = {} if cache else None

        part = read_participants(self.root)
        items = []
        for sub, f in scan_bids(self.root):
            m = part.get(sub)
            if not m:
                continue
            if target == "age":
                try:
                    label = float(m["age"])
                except (KeyError, ValueError):
                    continue
            elif target == "sex":
                s = m.get("sex", "")
                if s not in ("M", "F"):
                    continue
                label = 1.0 if s == "M" else 0.0
            else:
                raise ValueError(f"unknown target {target!r}")
            items.append((sub, f, label))

        if indices is not None:
            items = [items[i] for i in indices]
        self.items = items
        self.subjects = [s for s, _, _ in items]
        self.labels = np.array([l for _, _, l in items], dtype=np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sub, f, label = self.items[i]
        if self.cache is not None and i in self.cache:
            vol = self.cache[i]
        else:
            vol = to_volume(f, self.shape, fast=self.fast)
            if self.cache is not None:
                self.cache[i] = vol
        x = torch.from_numpy(vol.copy())
        if self.train:
            if torch.rand(1).item() < 0.5:      # brains are near-symmetric
                x = torch.flip(x, dims=[0])
            x = x + 0.02 * torch.randn_like(x)
        return x, torch.tensor(label, dtype=torch.float32)


def subject_split(dataset, val_fraction=0.25, seed=0):
    """Indices split BY SUBJECT, so no subject appears on both sides.

    Splitting by scan would put sub-01's two sessions in train and val and
    report memorisation as accuracy.
    """
    subs = np.array(dataset.subjects)
    uniq = np.unique(subs)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_fraction)))
    val = set(uniq[:n_val].tolist())
    is_val = np.array([s in val for s in subs])
    return np.where(~is_val)[0], np.where(is_val)[0]
