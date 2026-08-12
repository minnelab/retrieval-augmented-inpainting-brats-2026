"""Create fixed, committed train/val splits from the GLI training cases.

val_ids       : held-out evaluation set (never trained on)
val_small_ids : small subset of val for quick monitoring during training
train_ids     : everything else
"""
import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="splits")
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--n-val-small", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    root = Path(args.data_root)
    cases = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("BraTS")])
    print(f"{len(cases)} cases")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(cases))
    val_idx = set(perm[:args.n_val].tolist())
    val = [cases[i] for i in sorted(val_idx)]
    train = [cases[i] for i in range(len(cases)) if i not in val_idx]
    val_small = [val[i] for i in rng.permutation(len(val))[:args.n_val_small].tolist()]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "train_ids.txt").write_text("\n".join(train) + "\n")
    (out / "val_ids.txt").write_text("\n".join(val) + "\n")
    (out / "val_small_ids.txt").write_text("\n".join(sorted(val_small)) + "\n")
    print(f"train {len(train)}  val {len(val)}  val_small {len(val_small)} -> {out}")


if __name__ == "__main__":
    main()
