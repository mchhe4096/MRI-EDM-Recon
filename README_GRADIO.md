# MRI 重建封装（Gradio）

这是一个本地网页软件，用于单切片 MRI 重建。

## 1. 输入与输出

- 输入 `.pt`（两种格式都支持）：
  - 推荐：`k_us + mask`（真实欠采样输入）
  - 兼容：`kspace_full`（程序会按配置模拟欠采样）
- 输出：
  - `Recon` 重建图
  - `ZF` 零填充基线
  - `GT`（若输入含 `img_gt`）
  - `样本标准差图` 和 `多样本画廊`（用于展示多解能力）

## 1. 首次使用（初始化环境）

在项目根目录执行：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. 启动方式（推荐）

在项目根目录直接双击 `run.bat`。

等价命令行方式：

```bash
.\run.bat
```

## 3. 备用启动方式

```bash
python app.py
```

启动后打开：`http://127.0.0.1:7860`

## 4. 使用说明

- 默认 checkpoint 路径：`outputs/ckpts/last.pt`
- 若 checkpoint 没有 `ema_model`，取消勾选“使用 EMA 权重”
- 只有上传 `.pt` 内含 `img_gt` 时，才会显示 GT 与 PSNR
- 指标口径使用方式：
  - 勾选“显示当前项目口径指标”：显示 current PSNR
  - 勾选“显示论文基线口径指标”：显示论文基线 PSNR
  - 两个都勾选：同时显示两套口径

## 5. 关于 h5py

本封装软件只接收 `.pt` 输入，推理运行不依赖 `h5py`。  
只有在你需要把 `.h5` 数据预处理成 `.pt`（`scripts/data_process.py`）时，才需要安装 `h5py`。

