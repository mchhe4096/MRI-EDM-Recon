import argparse
from pathlib import Path

import numpy as np
import torch

from src.mri.operators import ifft2c


def normalize_kspace_by_mag_std(kspace: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, float]:
    """
    Single-scale normalization on k-space only.
    """
    scale_k = float(np.abs(kspace).std())
    k_norm = (kspace / (scale_k + eps)).astype(np.complex64)
    return k_norm, scale_k


def process_one_file(h5_path: Path, output_dir: Path, eps: float) -> int:
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py is required for data processing. Please install it first: pip install h5py") from e

    with h5py.File(h5_path, "r") as f:
        if "kspace" not in f:
            raise KeyError(f"'kspace' not found in {h5_path}")

        kspace = f["kspace"]  # expected [S,H,W] complex
        stem = h5_path.stem
        saved = 0

        for slice_id in range(kspace.shape[0]):  # pyright: ignore[reportAttributeAccessIssue]
            k_raw = np.asarray(kspace[slice_id])  # type: ignore[index]
            k_norm, scale_k = normalize_kspace_by_mag_std(k_raw, eps=eps)

            # [H,W] complex -> [2,H,W] float
            k_2hw = torch.view_as_real(torch.from_numpy(k_norm)).permute(2, 0, 1).contiguous().float()
            # Use the same centered-ortho operator as training pipeline.
            img_2hw = ifft2c(k_2hw.unsqueeze(0)).squeeze(0)

            # Save as [H,W,2] for compatibility with existing dataset loader.
            k_hwc2 = k_2hw.permute(1, 2, 0).contiguous()
            img_hwc2 = img_2hw.permute(1, 2, 0).contiguous()

            out_path = output_dir / f"{stem}_{slice_id:04d}.pt"
            torch.save(
                {
                    "kspace_full": k_hwc2,
                    "img_gt": img_hwc2,
                    "scale_k": scale_k,
                    "scale_mode": "kspace_mag_std_single",
                    "fname": h5_path.name,
                    "slice_id": int(slice_id),
                },
                out_path,
            )
            saved += 1

    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert raw h5 k-space data into unified-scale .pt slices.")
    p.add_argument(
        "--input_dir",
        type=str,
        default="/root/autodl-tmp/train",
        help="Directory with source .h5 files (e.g. /root/autodl-tmp/train or /root/autodl-tmp/val).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/train_v2",
        help="Directory to save converted .pt files (e.g. /root/autodl-tmp/train_v2 or /root/autodl-tmp/val_v2).",
    )
    p.add_argument("--glob", type=str, default="*.h5", help="Glob pattern for source files.")
    p.add_argument("--eps", type=float, default=1e-8, help="Numerical epsilon in normalization.")
    p.add_argument(
        "--clean_output",
        action="store_true",
        help="If set, remove existing .pt files in output_dir before processing.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    if args.clean_output:
        old_files = sorted(output_dir.glob("*.pt"))
        for p in old_files:
            p.unlink()
        print(f"[clean] removed {len(old_files)} old files from {output_dir}")

    h5_files = sorted(input_dir.glob(args.glob))
    if len(h5_files) == 0:
        raise FileNotFoundError(f"no files matched {args.glob} under {input_dir}")

    print(f"[start] input={input_dir} files={len(h5_files)} output={output_dir}")
    total_slices = 0
    for i, h5_path in enumerate(h5_files, start=1):
        n = process_one_file(h5_path, output_dir=output_dir, eps=float(args.eps))
        total_slices += n
        print(f"[{i:04d}/{len(h5_files):04d}] {h5_path.name}: {n} slices")

    print(f"[done] wrote {total_slices} slices to {output_dir}")


if __name__ == "__main__":
    main()
