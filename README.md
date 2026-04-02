# MRI-EDM-Recon

## 项目简介

本项目聚焦于**欠采样 MRI 重建**，基于扩散模型（Diffusion Model）进行图像重建。  
MRI 采集速度受限，常见加速方式是对 k-space 欠采样，但会引入伪影并增加重建难度。  
本项目结合扩散采样与 MRI 数据一致性约束，探索在欠采样条件下获得稳定、可对比的重建结果。

## 项目目标

- 构建面向欠采样 MRI 的扩散重建流程
- 结合 MRI 物理先验（如数据一致性）提升重建质量
- 支持多次采样与结果对比，分析重建稳定性

## 快速开始（Windows）

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\run.bat
```

启动后访问：`http://127.0.0.1:7860`

说明：
- 默认 checkpoint 路径为 `outputs/ckpts/last.pt`
- 输入为单切片 `.pt`，推荐包含 `k_us + mask`，也兼容 `kspace_full`

## 项目结构

```text
MRI-EDM-Recon/
├── app.py                       # Gradio 应用入口
├── services/                    # 封装推理逻辑与指标计算
├── src/                         # 核心算法代码（模型/采样/MRI 算子/训练）
├── scripts/                     # 训练、采样、数据处理脚本
├── Comparison/                  # 对比实验与口径相关脚本
├── requirements.txt             # 依赖清单
├── run.bat                      # Windows 一键启动脚本
└── README_GRADIO.md             # Gradio 使用说明（详细）
```

## 使用说明入口

- Gradio 封装使用文档：[`README_GRADIO.md`](README_GRADIO.md)
- 算法与实验代码入口：`src/`、`scripts/`、`Comparison/`
