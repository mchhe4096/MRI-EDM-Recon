# MRI-EDM-Recon

## 项目简介

本项目是本科大创项目，聚焦于**欠采样 MRI 重建**。  
磁共振成像（MRI）具有软组织高对比度和无辐射的优势，但采集速度较慢。常见加速策略是对 **K 空间**进行欠采样，但这会带来不适定重建问题。

现有主流方法通常依赖监督学习的深度神经网络，往往只给出单一重建结果，难以评估不确定性。  
本项目探索基于 **扩散模型（Diffusion Model）** 的 MRI 重建方法，从潜在分布中采样多个可能解，以提升重建结果的可靠性分析能力。

## 项目目标

- 构建面向欠采样 MRI 的扩散重建框架
- 结合 MRI 物理先验（如数据一致性）提升重建质量
- 支持多次采样与结果对比，分析重建不确定性

## 项目结构

```text
my-project/
├── src/                         # 核心源码
│   ├── data/                    # 数据集与预处理
│   │   ├── datasets.py
│   │   └── transforms.py
│   ├── diffusion/               # 扩散模型核心逻辑
│   │   ├── edm.py
│   │   ├── objective.py
│   │   └── sampler.py
│   ├── models/                  # 网络结构（UNet 及其变体）
│   │   ├── unet.py
│   │   └── unet_v2.py
│   ├── mri/                     # MRI 相关算子与约束
│   │   ├── conditioning.py
│   │   ├── data_consistency.py
│   │   ├── mask.py
│   │   └── operators.py
│   └── train/                   # 训练循环
│       └── train_loop.py
├── scripts/                     # 训练、采样、数据处理脚本
│   ├── data_process.py
│   ├── sample_param_search.py
│   ├── sample_val.py
│   └── train_steps.py
├── Comparison/                  # 对比实验与参数搜索相关代码
│   ├── sample_param_search_author.py
│   └── README.md
├── Comp_Fix/                    # 对比/复现实验的修正脚本
│   └── reval_tb.py
├── .gitignore
└── New.code-workspace
```

## 说明

- `src/` 是主要开发区域，包含模型、扩散过程、MRI 物理约束与训练流程。
- `scripts/` 提供实验入口脚本，便于快速训练和采样验证。
- `Comparison/` 与 `Comp_Fix/` 用于对比实验与复现实验修正。
