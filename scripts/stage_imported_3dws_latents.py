"""Validate and non-destructively stage imported-3DWS latent HDF5 files."""
import argparse, hashlib, json, os
from pathlib import Path
import h5py

p = argparse.ArgumentParser()
p.add_argument("--raw", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--training-manifest", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
records, frames = [], 0
for split in ("train", "val"):
    src_dir, dst_dir = a.raw / split, a.output / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.glob("episode_*_latents.h5")):
        with h5py.File(src, "r") as f:
            n = int(f["action"].shape[0])
            if f["action"].shape[1] != 14:
                raise ValueError(f"{src}: expected 14-D actions")
            if not all(k in f for k in ("z_bg", "z_mask", "z_off", "z_vals", "obj_occ")):
                raise ValueError(f"{src}: incomplete latent file")
        dst = dst_dir / src.name
        if dst.exists() or dst.is_symlink():
            if dst.resolve() != src.resolve():
                raise FileExistsError(f"refusing to replace {dst}")
        else:
            os.symlink(src.resolve(), dst)
        frames += n
        records.append(dict(split=split, file=str(dst), frames=n, bytes=src.stat().st_size))
manifest = dict(
    format="imported_3dws_latents_v1",
    source_manifest=str(a.training_manifest),
    episodes=len(records), frames=frames,
    train=sum(r["split"] == "train" for r in records),
    val=sum(r["split"] == "val" for r in records),
    bytes=sum(r["bytes"] for r in records), records=records,
)
tmp = a.output / f"latent_manifest.json.tmp.{os.getpid()}"
tmp.write_text(json.dumps(manifest, indent=2) + "\n")
os.replace(tmp, a.output / "latent_manifest.json")
print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2))
