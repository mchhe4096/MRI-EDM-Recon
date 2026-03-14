import os, h5py, torch
import numpy as np

def fft2c(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def ifft2c(kspace):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))

def normalize_by_mag_std(x, eps=1e-8):
    mag = np.abs(x)
    s = mag.std()
    return x / (s + eps), float(s)

out_dir = "./data/train/"
inp_dir = "./T1/train/"
os.makedirs(out_dir, exist_ok=True)

for file in os.listdir(inp_dir):
    with h5py.File(os.path.join(inp_dir, file), "r") as f:
        kspace = f["kspace"]  # [S,H,W] complex-ish
        for i in range(kspace.shape[0]): # pyright: ignore[reportAttributeAccessIssue]
            k = kspace[i] # type: ignore
            k_norm, s_k = normalize_by_mag_std(k)

            img = ifft2c(k_norm)
            img_norm, s_img = normalize_by_mag_std(img)

            # 建议：统一只用一种 scale（比如用 s_k），别对 k 和 img 各归一化一次
            # 这里先保留两者，便于你对齐旧数据
            torch.save(
                {
                    "kspace_full": torch.view_as_real(torch.from_numpy(k_norm.astype(np.complex64))).float(), # [H,W,2]
                    "img_gt": torch.view_as_real(torch.from_numpy(img_norm.astype(np.complex64))).float(),     # [H,W,2]
                    "scale_k": s_k,
                    "scale_img": s_img,
                    "fname": file,
                    "slice_id": i,
                },
                os.path.join(out_dir, f"{file.replace('.h5','')}_{i:04d}.pt"),
            )
