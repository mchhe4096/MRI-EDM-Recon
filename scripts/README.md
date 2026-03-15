# Scripts Usage

本目录当前推荐只使用以下脚本：
- `train_steps.py`：正式训练/续训
- `sample_val.py`：验证集采样与可视化
- `data_process.py`：原始数据预处理

`train_debug.py`：不再使用。

## 运行前说明

- 在项目根目录执行：`/root/autodl-tmp/my-project`
- 建议使用模块方式启动，避免 `No module named src`
  - `python -m scripts.train_steps`
  - `python -m scripts.sample_val`

## 1. 正式训练

```bash
cd /root/autodl-tmp/my-project
python -m scripts.train_steps \
  --train_root /root/autodl-tmp/train_v2 \
  --val_root /root/autodl-tmp/val_v2 \
  --include_mask_channel \
  --sigma_data 0.5 \
  --steps 100000 \
  --batch 1 \
  --num_workers 2 \
  --outdir ../outputs/ckpts
```

## 2. 断点续训

```bash
cd /root/autodl-tmp/my-project
python -m scripts.train_steps \
  --train_root /root/autodl-tmp/train_v2 \
  --val_root /root/autodl-tmp/val_v2 \
  --include_mask_channel \
  --sigma_data 0.5 \
  --steps 100000 \
  --batch 1 \
  --num_workers 2 \
  --resume ../outputs/ckpts/last.pt \
  --outdir ../outputs/ckpts
```

## 3. 采样验证（不带 DC）

```bash
cd /root/autodl-tmp/my-project
python -m scripts.sample_val \
  --ckpt ../outputs/ckpts/last.pt \
  --val_root /root/autodl-tmp/val_v2 \
  --include_mask_channel \
  --num_cases 8 \
  --num_samples 1 \
  --steps 40 \
  --outdir ../outputs/samples_no_dc
```

说明：
- `--num_cases 0` 表示对 `val_root` 下全部样本采样。
- 默认会保存每个 case 的 `result.pt`（便于后续 PSNR/SSIM 等评估）。
- 若需节省空间，保持默认（不加 `--save_all_samples_pt`）；仅保存 mean/std/target 等统计量。
- 若需不确定性或多样本后处理，可加 `--save_all_samples_pt` 保存全部采样张量。

## 4. 采样验证（带 DC，推荐）

```bash
cd /root/autodl-tmp/my-project
python -m scripts.sample_val \
  --ckpt ../outputs/ckpts/last.pt \
  --val_root /root/autodl-tmp/val_v2 \
  --include_mask_channel \
  --num_cases 0 \
  --num_samples 5 \
  --steps 40 \
  --dc --dc_start 0.6 --dc_every 2 --dc_lam 0.15 --dc_ramp \
  --outdir ../outputs/samples_dc
```

补充：
- 训练脚本使用 EDM 预条件目标，并支持 `--sigma_data`（默认 `0.5`）。
- 采样脚本会优先读取 checkpoint 中的 `sigma_data`；不再支持旧 checkpoint 格式。

## 5. 数据预处理

```bash
cd /root/autodl-tmp/my-project
# 1) 重新导出 train（统一尺度：只对 kspace 做归一化）
python -m scripts.data_process \
  --input_dir /root/autodl-tmp/train \
  --output_dir /root/autodl-tmp/train_v2 \
  --clean_output

# 2) 重新导出 val
python -m scripts.data_process \
  --input_dir /root/autodl-tmp/val \
  --output_dir /root/autodl-tmp/val_v2 \
  --clean_output
```

说明：
- 这一步会生成统一尺度数据（`scale_mode=kspace_mag_std_single`），需要重跑后再训练。
- 新目录与原始 `train` / `val` 同级：`/root/autodl-tmp/train_v2`、`/root/autodl-tmp/val_v2`。
- 当前数据集按 `--accel` 和 `--center_frac` 动态生成 mask。
