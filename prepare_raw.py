#!/usr/bin/env python3
"""Turn a raw BIDS dataset into 3D volumes a CNN can train on.

Unlike the flattened `train_data.npy` these sites currently serve (50k random
features per subject, no spatial structure), this keeps the image as an image:
one downsampled 3D volume per scan, so a convolutional model has something to
convolve over.

Writes, into the target dataset directory:
    images.npy   float32 (N, D, H, W)   brain volumes, intensity-normalised
    targets.npy  float32 (N,)           age in years
    sex.npy      int64   (N,)           0=F 1=M
    subjects.npy <U16    (N,)           participant_id, for subject-wise splits
    dataset_metadata.json
"""
import argparse, csv, json, pathlib, sys
import numpy as np
import nibabel as nib

def find_scans(root: pathlib.Path):
    """Every real T1w under sub-*/anat or sub-*/ses-*/anat, keyed by subject."""
    out = []
    for pat in ("sub-*/anat/*T1w.nii.gz", "sub-*/ses-*/anat/*T1w.nii.gz"):
        for f in sorted(root.glob(pat)):
            # git-annex datasets leave dangling symlinks when content was never
            # fetched; they glob fine and fail on open. Skip them explicitly.
            if not f.is_file():
                continue
            sub = next(p for p in f.parts if p.startswith("sub-"))
            out.append((sub, f))
    return out

def read_participants(root: pathlib.Path):
    """participant_id -> {age, sex}. Handles the BOM some BIDS files carry."""
    meta = {}
    tsv = root / "participants.tsv"
    if not tsv.is_file():
        return meta
    with tsv.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            pid = (r.get("participant_id") or "").strip()
            if not pid:
                continue
            # Some datasets name the column `sex`, others `gender`.
            meta[pid] = {
                "age": (r.get("age") or "").strip(),
                "sex": (r.get("sex") or r.get("gender") or "").strip().upper(),
            }
    return meta

def load_volume(path: pathlib.Path, out_shape):
    """Load one scan, crop the empty margin, resize, normalise to zero mean."""
    vol = np.asanyarray(nib.load(str(path)).dataobj, dtype=np.float32)

    # Robust intensity scaling: clip at the 99.5th percentile so a few bright
    # voxels do not compress the rest of the range.
    hi = np.percentile(vol, 99.5)
    if hi > 0:
        vol = np.clip(vol, 0, hi) / hi

    # Crop to the bounding box of non-background voxels, so subjects with
    # different amounts of empty space around the head end up comparable.
    mask = vol > 0.1
    if mask.any():
        idx = np.array(np.nonzero(mask))
        lo_c, hi_c = idx.min(axis=1), idx.max(axis=1) + 1
        vol = vol[lo_c[0]:hi_c[0], lo_c[1]:hi_c[1], lo_c[2]:hi_c[2]]

    vol = resize_linear(vol, out_shape)

    # Per-scan standardisation over brain voxels only: scanner gain differs
    # between sites, and this is what keeps a federated model from learning
    # "which site" instead of "what age".
    brain = vol[vol > 0.01]
    if brain.size:
        vol = (vol - brain.mean()) / (brain.std() + 1e-6)
    return vol.astype(np.float32)

def resize_linear(vol, out_shape):
    """Trilinear resize with no SciPy dependency (torch is already required)."""
    import torch
    t = torch.from_numpy(vol)[None, None]
    t = torch.nn.functional.interpolate(
        t, size=tuple(out_shape), mode="trilinear", align_corners=False)
    return t[0, 0].numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="raw BIDS dataset root")
    ap.add_argument("target", help="output directory")
    ap.add_argument("--shape", default="64,64,64",
                    help="output volume D,H,W (default 64,64,64)")
    a = ap.parse_args()

    src, dst = pathlib.Path(a.source), pathlib.Path(a.target)
    shape = tuple(int(x) for x in a.shape.split(","))
    dst.mkdir(parents=True, exist_ok=True)

    scans = find_scans(src)
    if not scans:
        # Fail loudly rather than writing an empty dataset the server would
        # happily register and training would then find nothing in.
        sys.exit(f"ERROR: no readable T1w images under {src}. If this is a "
                 f"DataLad/git-annex dataset, the content was never fetched "
                 f"(`datalad get`), and the symlinks are dangling.")

    part = read_participants(src)
    imgs, ages, sexes, subs, skipped = [], [], [], [], []
    for sub, f in scans:
        m = part.get(sub, {})
        try:
            age = float(m.get("age", ""))
        except ValueError:
            skipped.append((sub, "no age"))
            continue
        try:
            imgs.append(load_volume(f, shape))
        except Exception as exc:
            skipped.append((sub, f"unreadable: {exc}"))
            continue
        ages.append(age)
        sexes.append(1 if m.get("sex", "").startswith("M") else 0)
        subs.append(sub)
        print(f"  {sub}  {f.name}  age={age:g}")

    if not imgs:
        sys.exit("ERROR: no scans had a usable age in participants.tsv.")

    X = np.stack(imgs).astype(np.float32)
    y = np.asarray(ages, dtype=np.float32)
    np.save(dst / "images.npy", X)
    np.save(dst / "targets.npy", y)
    np.save(dst / "sex.npy", np.asarray(sexes, dtype=np.int64))
    np.save(dst / "subjects.npy", np.asarray(subs))

    meta = {
        "dataset": src.name,
        "n_scans": int(X.shape[0]),
        "n_subjects": int(len(set(subs))),
        "volume_shape": list(shape),
        "target": "age (years)",
        "age_min": float(y.min()), "age_max": float(y.max()),
        "age_mean": float(y.mean()), "age_std": float(y.std()),
        "skipped": skipped,
        "note": "3D T1w volumes, cropped to brain bbox, resized, z-scored per scan.",
    }
    (dst / "dataset_metadata.json").write_text(json.dumps(meta, indent=2))
    (dst / "dataset_description.json").write_text(json.dumps(
        {"Name": dst.name, "BIDSVersion": "1.8.0", "DatasetType": "derivative"}, indent=2))
    print(f"\nwrote {X.shape[0]} volumes {X.shape[1:]} -> {dst}")
    print(f"age {y.min():.0f}-{y.max():.0f} mean {y.mean():.1f}")
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}")

if __name__ == "__main__":
    main()
