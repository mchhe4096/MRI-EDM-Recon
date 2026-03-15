# Scripts Usage

本目录当前推荐脚本：
- `train_steps.py`：测试训练 / 正式训练 / 续训
- `sample_val.py`：验证集采样与可视化
- `data_process.py`：原始 `.h5` 预处理为切片 `.pt`

## 运行前说明

- 在项目根目录执行：`/root/autodl-tmp/my-project`
- 建议模块方式启动，避免 `No module named src`
- 除“测试训练”外，下面命令统一使用 `nohup` 后台跑，并记录带时间戳日志

## 1. 数据预处理（nohup 后台 + 日志）

```bash
cd /root/autodl-tmp/my-project
mkdir -p ../outputs/logs

# 1) train -> train_v2
ts=$(date +%Y%m%d_%H%M%S)
nohup python -m scripts.data_process \
  --input_dir /root/autodl-tmp/train \
  --output_dir /root/autodl-tmp/train_v2 \
  --clean_output \
  > ../outputs/logs/data_process_train_${ts}.log 2>&1 &
echo "pid=$! log=../outputs/logs/data_process_train_${ts}.log"

# 2) val -> val_v2
ts=$(date +%Y%m%d_%H%M%S)
nohup python -m scripts.data_process \
  --input_dir /root/autodl-tmp/val \
  --output_dir /root/autodl-tmp/val_v2 \
  --clean_output \
  > ../outputs/logs/data_process_val_${ts}.log 2>&1 &
echo "pid=$! log=../outputs/logs/data_process_val_${ts}.log"
```

说明：
- 新目录与原始 `train` / `val` 同级：`/root/autodl-tmp/train_v2`、`/root/autodl-tmp/val_v2`。
- 建议先确认 train 预处理完成，再启动 val 或训练。

## 2. 测试训练（前台跑，不用 nohup）

```bash
cd /root/autodl-tmp/my-project
python -m scripts.train_steps \
  --train_root /root/autodl-tmp/train_v2 \
  --val_root /root/autodl-tmp/val_v2 \
  --include_mask_channel \
  --sigma_data 0.5 \
  --warmup_steps 50 \
  --min_lr_ratio 0.2 \
  --ema_decay 0.999 \
  --steps 200 \
  --train_subset 256 \
  --val_subset 32 \
  --batch 2 \
  --num_workers 2 \
  --log_every 20 \
  --val_every 100 \
  --ckpt_every 200 \
  --outdir ../outputs/ckpts_smoke
```

## 3. 正式训练（nohup 后台 + 日志）

48GB 显存推荐（实测接近吃满）：
- `base_ch=128, batch=28`：峰值约 `46.17GB`
- 若 OOM，依次降到 `batch=24` 或 `batch=20`

```bash
cd /root/autodl-tmp/my-project
mkdir -p ../outputs/logs
ts=$(date +%Y%m%d_%H%M%S)
nohup python -m scripts.train_steps \
  --train_root /root/autodl-tmp/train_v2 \
  --val_root /root/autodl-tmp/val_v2 \
  --include_mask_channel \
  --base_ch 128 \
  --sigma_data 0.5 \
  --warmup_steps 2000 \
  --min_lr_ratio 0.05 \
  --ema_decay 0.9999 \
  --steps 100000 \
  --batch 24 \
  --num_workers 8 \
  --outdir ../outputs/ckpts \
  > ../outputs/logs/train_${ts}.log 2>&1 &
echo "pid=$! log=../outputs/logs/train_${ts}.log"
```

## 4. 断点续训（nohup 后台 + 日志）

```bash
cd /root/autodl-tmp/my-project
mkdir -p ../outputs/logs
ts=$(date +%Y%m%d_%H%M%S)
nohup python -m scripts.train_steps \
  --train_root /root/autodl-tmp/train_v2 \
  --val_root /root/autodl-tmp/val_v2 \
  --include_mask_channel \
  --base_ch 128 \
  --sigma_data 0.5 \
  --warmup_steps 2000 \
  --min_lr_ratio 0.05 \
  --ema_decay 0.9999 \
  --steps 100000 \
  --batch 24 \
  --num_workers 8 \
  --resume ../outputs/ckpts/last.pt \
  --outdir ../outputs/ckpts \
  > ../outputs/logs/resume_${ts}.log 2>&1 &
echo "pid=$! log=../outputs/logs/resume_${ts}.log"
```

## 5. 采样验证（nohup 后台 + 日志）

### 5.1 不带 DC

```bash
cd /root/autodl-tmp/my-project
mkdir -p ../outputs/logs
ts=$(date +%Y%m%d_%H%M%S)
nohup python -m scripts.sample_val \
  --ckpt ../outputs/ckpts/last.pt \
  --val_root /root/autodl-tmp/val_v2 \
  --include_mask_channel \
  --num_cases 8 \
  --num_samples 1 \
  --steps 40 \
  --outdir ../outputs/samples_no_dc \
  > ../outputs/logs/sample_no_dc_${ts}.log 2>&1 &
echo "pid=$! log=../outputs/logs/sample_no_dc_${ts}.log"
```

### 5.2 带 DC（推荐）

```bash
cd /root/autodl-tmp/my-project
mkdir -p ../outputs/logs
ts=$(date +%Y%m%d_%H%M%S)
nohup python -m scripts.sample_val \
  --ckpt ../outputs/ckpts/last.pt \
  --val_root /root/autodl-tmp/val_v2 \
  --include_mask_channel \
  --num_cases 0 \
  --num_samples 5 \
  --steps 40 \
  --dc --dc_start 0.6 --dc_every 2 --dc_lam 0.15 --dc_ramp \
  --outdir ../outputs/samples_dc \
  > ../outputs/logs/sample_dc_${ts}.log 2>&1 &
echo "pid=$! log=../outputs/logs/sample_dc_${ts}.log"
```

## 补充

- 训练脚本使用 EDM 预条件目标，并支持 `--sigma_data`（默认 `0.5`）。
- 训练默认启用 EMA（checkpoint 含 `ema_model`）和 warmup+cosine 学习率调度。
- 采样默认使用 EMA 权重（可用 `--no-use_ema` 切回原始 `model` 权重）。
- 旧架构 checkpoint 与当前 `UNetV2` 不兼容，不能直接 `--resume`。
