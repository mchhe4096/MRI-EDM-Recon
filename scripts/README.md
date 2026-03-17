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

参数注释（默认值 / 资源压力）：
- `--input_dir`：原始 `.h5` 输入目录；默认 `/root/autodl-tmp/train`；压力：磁盘读取 I/O。
- `--output_dir`：转换后 `.pt` 输出目录；默认 `/root/autodl-tmp/train_v2`；压力：磁盘写入 I/O 与容量。
- `--clean_output`：清空输出目录已有 `.pt` 后再处理；默认 `False`；压力：额外磁盘删除操作（不可恢复，请谨慎）。
- `--glob`：输入文件匹配模式；默认 `*.h5`；压力：仅影响扫描数量。
- `--eps`：归一化数值稳定项；默认 `1e-8`；压力：无硬件压力。

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

参数注释（默认值 / 资源压力）：
- `--train_root`：训练集目录；默认 `../data/train`；压力：磁盘读取 I/O。
- `--val_root`：验证集目录；默认 `../data/val`；压力：磁盘读取 I/O。
- `--include_mask_channel`：将 mask 作为 cond 第 3 通道；默认 `False`；压力：显存/算力小幅增加。
- `--sigma_data`：EDM 预条件参数；默认 `0.5`；压力：无直接硬件压力（主要影响收敛行为）。
- `--warmup_steps`：学习率 warmup 步数；默认 `2000`；压力：无直接硬件压力。
- `--min_lr_ratio`：余弦退火最小学习率比例；默认 `0.05`；压力：无直接硬件压力。
- `--ema_decay`：EMA 衰减；默认 `0.9999`；压力：EMA 本身常驻，参数仅影响更新速度。
- `--steps`：总训练步数；默认 `100000`；压力：线性增加总训练时长。
- `--train_subset`：训练子集样本数，`0` 表示全量；默认 `0`；压力：减小可显著降时间与 I/O。
- `--val_subset`：验证子集样本数，`0` 表示全量；默认 `0`；压力：减小可显著降验证开销。
- `--batch`：训练 batch；默认 `1`；压力：显存主开关（近线性增长）。
- `--num_workers`：DataLoader worker 数；默认 `0`；压力：CPU 与内存占用上升，I/O 吞吐提升。
- `--log_every`：日志打印间隔；默认 `50`；压力：很小（日志 I/O）。
- `--val_every`：验证间隔步数；默认 `1000`；压力：值越小验证越频繁，训练吞吐下降。
- `--ckpt_every`：checkpoint 保存间隔；默认 `2000`；压力：磁盘写入与容量占用。
- `--outdir`：checkpoint 输出目录；默认 `../outputs/ckpts`；压力：磁盘容量占用。

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

参数注释（默认值 / 资源压力）：
- `--base_ch`：UNet 基础通道数；默认 `128`；压力：显存与算力大幅增加（关键开关）。
- `--batch`：训练 batch；默认 `1`；压力：显存主开关（OOM 优先先降这个）。
- `--num_workers`：DataLoader worker 数；默认 `0`；压力：CPU 与内存占用上升。
- `--steps`：总训练步数；默认 `100000`；压力：线性增加总训练时长。
- `--warmup_steps`：warmup 步数；默认 `2000`；压力：无直接硬件压力。
- `--min_lr_ratio`：学习率最小比例；默认 `0.05`；压力：无直接硬件压力。
- `--ema_decay`：EMA 衰减；默认 `0.9999`；压力：无额外硬件变化。
- `--sigma_data`：EDM 预条件参数；默认 `0.5`；压力：无直接硬件压力。
- `--train_root`：训练集目录；默认 `../data/train`；压力：磁盘读取 I/O。
- `--val_root`：验证集目录；默认 `../data/val`；压力：磁盘读取 I/O。
- `--include_mask_channel`：是否拼接 mask 通道；默认 `False`；压力：显存/算力小幅增加。
- `--outdir`：checkpoint 输出目录；默认 `../outputs/ckpts`；压力：磁盘容量占用。

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

参数注释（默认值 / 资源压力）：
- `--resume`：续训 checkpoint 路径；默认 `""`（不续训）；压力：启动时额外一次大文件读取。
- `--base_ch`：UNet 基础通道数；默认 `128`；压力：显存与算力大幅增加。
- `--batch`：训练 batch；默认 `1`；压力：显存主开关。
- `--num_workers`：DataLoader worker 数；默认 `0`；压力：CPU/内存占用上升。
- `--steps`：目标总步数；默认 `100000`；压力：线性增加训练时长。
- `--warmup_steps`：warmup 步数；默认 `2000`；压力：无直接硬件压力。
- `--min_lr_ratio`：学习率最小比例；默认 `0.05`；压力：无直接硬件压力。
- `--ema_decay`：EMA 衰减；默认 `0.9999`；压力：无额外硬件变化。
- `--sigma_data`：EDM 预条件参数；默认 `0.5`；压力：无直接硬件压力。
- `--train_root`：训练集目录；默认 `../data/train`；压力：磁盘读取 I/O。
- `--val_root`：验证集目录；默认 `../data/val`；压力：磁盘读取 I/O。
- `--include_mask_channel`：是否拼接 mask 通道；默认 `False`；压力：显存/算力小幅增加。
- `--outdir`：checkpoint 输出目录；默认 `../outputs/ckpts`；压力：磁盘容量占用。

## 5. 参数搜索（sample_param_search，nohup 后台 + 日志）

```bash
cd /root/autodl-tmp/my-project
mkdir -p ../outputs/logs
ts=$(date +%Y%m%d_%H%M%S)
nohup python -m scripts.sample_param_search \
  --ckpt ../outputs/ckpts/last.pt \
  --val_root /root/autodl-tmp/val_v2 \
  --num_cases 0 \
  --num_samples 1 \
  --init_mode_list zf \
  --steps_list 30,40 \
  --dc_start_list 0.6,0.8 \
  --dc_every_list 2,3 \
  --dc_lam_list 0.08,0.12,0.15 \
  --dc_ramp_list 0,1 \
  --include_no_dc_baseline \
  --rank_metric psnr \
  --outdir ../outputs/param_search \
  > ../outputs/logs/param_search_${ts}.log 2>&1 &
echo "pid=$! log=../outputs/logs/param_search_${ts}.log"
```

参数注释（默认值 / 资源压力）：
- `--ckpt`：采样权重路径；默认 `../outputs/ckpts/last.pt`；压力：启动时读取大文件。
- `--val_root`：验证集目录；默认 `/root/autodl-tmp/val_v2`；压力：磁盘读取 I/O。
- `--num_cases`：参与搜索的 case 数，`0` 表示全量；默认 `0`；压力：线性增加总耗时。
- `--num_samples`：每个 case 每组参数采样次数；默认 `1`；压力：线性增加 GPU 耗时。
- `--init_mode_list`：采样初始化模式候选（`noise,zf,blend`）；默认 `zf`；压力：几乎无硬件差异（主要影响重建稳定性）。
- `--init_blend_list`：`blend` 模式的 alpha 候选；默认 `0.5`；压力：几乎无硬件差异。
- `--steps_list`：采样步数候选；默认 `40`；压力：每个样本耗时近线性增长。
- `--dc_start_list`：DC 开始比例候选；默认 `0.6,0.8`；压力：影响 DC 调用次数。
- `--dc_every_list`：DC 间隔候选；默认 `2,3`；压力：值越小 DC 次数越多，耗时上升。
- `--dc_lam_list`：DC 强度候选；默认 `0.08,0.12,0.15`；压力：几乎无硬件差异（主要影响结果）。
- `--dc_ramp_list`：是否末段增大 DC 强度候选；默认 `0,1`；压力：几乎无硬件差异。
- `--include_no_dc_baseline`：是否加入 no-DC 基线；默认 `True`；压力：会额外增加一组配置计算量。
- `--rank_metric`：TopK 排序指标；默认 `psnr`；压力：无硬件压力。
- `--outdir`：搜索结果输出目录；默认 `../outputs/param_search`；压力：磁盘写入与容量占用。

说明：
- 输出文件在 `../outputs/param_search/`，包括 `summary.csv`、`per_case.csv`、`topk.json`、`meta.json`。
- 脚本会额外输出 `baseline_zf.json` 与 `baseline_zf_per_case.csv`，用于核对“是否真正超过 zf 基线”。
- 若需要真实 LPIPS，请先安装：`pip install lpips`。

## 6. 采样验证（nohup 后台 + 日志）

### 6.1 不带 DC

```bash
cd /root/autodl-tmp/my-project
mkdir -p ../outputs/logs
ts=$(date +%Y%m%d_%H%M%S)
nohup python -m scripts.sample_val \
  --ckpt ../outputs/ckpts/last.pt \
  --val_root /root/autodl-tmp/val_v2 \
  --init_mode zf \
  --num_cases 8 \
  --num_samples 1 \
  --steps 40 \
  --outdir ../outputs/samples_no_dc \
  > ../outputs/logs/sample_no_dc_${ts}.log 2>&1 &
echo "pid=$! log=../outputs/logs/sample_no_dc_${ts}.log"
```

参数注释（默认值 / 资源压力）：
- `--ckpt`：采样权重路径；默认 `../outputs/ckpts/last.pt`；压力：启动时读取大文件。
- `--val_root`：验证集目录；默认 `../data/val`；压力：磁盘读取 I/O。
- `--include_mask_channel`：cond 是否拼接 mask；默认 `None`（自动继承 ckpt 训练配置）；压力：显存/算力小幅增加。
- `--num_cases`：采样 case 数，`0` 表示全量；默认 `8`；压力：线性增加总耗时。
- `--num_samples`：每个 case 采样次数；默认 `8`；压力：线性增加 GPU 耗时。
- `--init_mode`：采样初始化模式（`noise/zf/blend`）；默认 `zf`；压力：几乎无硬件差异（主要影响重建质量）。
- `--init_blend`：`blend` 模式 alpha；默认 `0.5`；压力：几乎无硬件差异。
- `--steps`：采样步数；默认 `40`；压力：每样本耗时近线性增长。
- `--outdir`：输出目录；默认 `../outputs/samples`；压力：磁盘写入与容量占用。
- `--use_ema`：是否使用 EMA 权重；默认 `True`；压力：无硬件压力（主要影响效果稳定性）。

### 6.2 带 DC（推荐）

```bash
cd /root/autodl-tmp/my-project
mkdir -p ../outputs/logs
ts=$(date +%Y%m%d_%H%M%S)
nohup python -m scripts.sample_val \
  --ckpt ../outputs/ckpts/last.pt \
  --val_root /root/autodl-tmp/val_v2 \
  --init_mode zf \
  --num_cases 0 \
  --num_samples 5 \
  --steps 40 \
  --dc --dc_start 0.6 --dc_every 2 --dc_lam 0.15 --dc_ramp \
  --outdir ../outputs/samples_dc \
  > ../outputs/logs/sample_dc_${ts}.log 2>&1 &
echo "pid=$! log=../outputs/logs/sample_dc_${ts}.log"
```

参数注释（默认值 / 资源压力）：
- `--ckpt`：采样权重路径；默认 `../outputs/ckpts/last.pt`；压力：启动时读取大文件。
- `--val_root`：验证集目录；默认 `../data/val`；压力：磁盘读取 I/O。
- `--include_mask_channel`：cond 是否拼接 mask；默认 `None`（自动继承 ckpt 训练配置）；压力：显存/算力小幅增加。
- `--num_cases`：采样 case 数，`0` 表示全量；默认 `8`；压力：线性增加总耗时。
- `--dc`：启用 DC；默认 `False`；压力：每步增加 FFT/IFFT，GPU 耗时上升。
- `--dc_start`：从总步数的哪个比例后开始 DC；默认 `0.6`；压力：越早开始，DC 次数越多。
- `--dc_every`：DC 间隔步数；默认 `2`；压力：越小越耗时。
- `--dc_lam`：DC 强度；默认 `0.15`；压力：几乎无硬件差异（主要影响重建质量与稳定性）。
- `--dc_ramp`：末段是否逐步增大 DC 强度；默认 `False`；压力：几乎无硬件差异。
- `--init_mode`：采样初始化模式（`noise/zf/blend`）；默认 `zf`；压力：几乎无硬件差异（主要影响重建质量）。
- `--init_blend`：`blend` 模式 alpha；默认 `0.5`；压力：几乎无硬件差异。
- `--num_samples`：每 case 采样次数；默认 `8`；压力：线性增加 GPU 耗时。
- `--steps`：采样步数；默认 `40`；压力：每样本耗时近线性增长。
- `--outdir`：输出目录；默认 `../outputs/samples`；压力：磁盘写入与容量占用。
- `--use_ema`：是否使用 EMA 权重；默认 `True`；压力：无硬件压力。

## 补充

- 训练脚本使用 EDM 预条件目标，并支持 `--sigma_data`（默认 `0.5`）。
- 训练默认启用 EMA（checkpoint 含 `ema_model`）和 warmup+cosine 学习率调度。
- 采样默认使用 EMA 权重（可用 `--no-use_ema` 切回原始 `model` 权重）。
- 采样默认初始化模式为 `zf`；如需复现实验旧行为，可设 `--init_mode noise`。
- 旧架构 checkpoint 与当前 `UNetV2` 不兼容，不能直接 `--resume`。
- 若需修正历史 TensorBoard 验证曲线（不影响当前训练），可运行 `Comp_Fix/reval_tb.py`：
```bash
cd /root/autodl-tmp/my-project
mkdir -p ../outputs/logs
ts=$(date +%Y%m%d_%H%M%S)
nohup python Comp_Fix/reval_tb.py \
  --ckpt_dir ../outputs/ckpts \
  --val_root /root/autodl-tmp/val_v2 \
  --use_ema \
  --tb_root /root/tf-logs \
  --run_name reval_${ts} \
  > ../outputs/logs/reval_${ts}.log 2>&1 &
echo "pid=$! log=../outputs/logs/reval_${ts}.log"
```

参数注释（默认值 / 资源压力）：
- `--ckpt_dir`：checkpoint 目录；默认 `../outputs/ckpts`；压力：大量 checkpoint 时读取耗时上升。
- `--val_root`：验证集目录；默认 `/root/autodl-tmp/val_v2`；压力：磁盘读取 I/O。
- `--use_ema`：是否优先评估 EMA 权重；默认 `True`；压力：无硬件压力。
- `--tb_root`：TensorBoard 输出根目录；默认 `/root/tf-logs`；压力：磁盘写入。
- `--run_name`：TensorBoard run 名称；默认 `""`（脚本自动生成）；压力：无硬件压力。
- 未显式设置参数会使用脚本默认值：`batch=1`、`num_workers=0`、`max_batches=20`、`sigma_proxy=1.0`、`device=自动选择`。
