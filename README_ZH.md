<div align="center">

# Awesome Adverse Weather Image Restoration

### 论文 *Deep Image Restoration in Adverse Weather: A Survey* 官方配套仓库

Zhenbo Song, Ruixin Li<sup>\*</sup>, Zhenyuan Zhang, Tao Wang, Jianfeng Lu, Xin Yu, Kaihao Zhang<sup>†</sup>

<sup>\*</sup> 共同一作 · <sup>†</sup> 通讯作者

[![Paper](https://img.shields.io/badge/Paper-Neural%20Networks-1f6feb)](https://doi.org/10.1016/j.neunet.2026.109472)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.neunet.2026.109472-blue)](https://doi.org/10.1016/j.neunet.2026.109472)
[![Open Access](https://img.shields.io/badge/Open%20Access-CC%20BY%204.0-green)](https://creativecommons.org/licenses/by/4.0/)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[English](README.md) | [简体中文](README_ZH.md)

</div>

<p align="center">
  <img src="assets/images/adverse_weather_examples.png" width="900" alt="不良天气退化示例(论文 Fig. 1)">
</p>

## 🔥 动态

- **2026.09** — 发布本仓库作为论文配套资源,持续整理不良天气图像恢复的方法、数据集、损失函数、评价指标与 benchmark 结果。
- **2026.08** — 论文被 *Neural Networks* 接收并在线发表(2026 年 8 月 7 日上线)。

## 🌟 简介

不良天气图像恢复旨在从雾、雨、雪等天气退化图像中恢复干净的背景场景。随着深度学习的发展,面向特定天气类型的**单任务恢复**取得了显著进展;近年来,在统一框架内处理多种退化的 **All-in-One (AiO) 方法**也迅速兴起。

该综述从**网络架构与学习范式**两个视角统一梳理单任务与 AiO 恢复模型,并进一步综述了数据集、损失函数、评价指标与代表性 benchmark 结果,最后讨论关键挑战与未来方向。

本仓库是持续维护的论文配套资源,结构完全对应论文组织方式,展示风格参考了 [Awesome-Image-Restoration](https://github.com/TaoWangzj/Awesome-Image-Restoration)。

### 主要范围

1. **单任务(退化特定)恢复** — 去雾、去雨、去雪、其他天气退化与视频恢复。
2. **网络架构演进** — CNN、GAN、Transformer/Mamba 与扩散模型。
3. **学习范式** — 有监督 (SL)、无监督 (UL)、半监督 (SSL)、弱监督 (WSL)。
4. **All-in-One 恢复** — One-to-One 与 One-to-Many 建模(MoE、prompt、条件路由、扩散),以及通用型 AiO 模型。
5. **评测** — 数据集、损失函数、全参考/无参考指标,以及各任务 benchmark 结果。

> **内容溯源政策。** 本仓库所有条目均可在论文中溯源:方法与数据集表逐行复刻论文总览表 (Tables 2–7),benchmark 数值来自论文 benchmark 表 (Tables 8–13),分组结构对应论文相应章节。条目按论文原样引用(包括 "ACMMM 2024" 这类缩写会议名,以及论文只以 作者-年份 引注形式给出、未给出方法名的条目)。

<p align="center">
  <img src="assets/images/survey_overview.png" width="1000" alt="论文组织结构总览(论文 Fig. 2)">
</p>

## 🔮 仓库内容

- [1. 退化特定图像恢复](#1-退化特定图像恢复)
  - [1.1 图像去雾](#11-图像去雾)
  - [1.2 图像去雨](#12-图像去雨)
  - [1.3 图像去雪](#13-图像去雪)
  - [1.4 其他不良天气恢复任务](#14-其他不良天气恢复任务)
  - [1.5 基于视频的不良天气恢复](#15-基于视频的不良天气恢复)
- [2. All-in-One 图像恢复](#2-all-in-one-图像恢复)
  - [2.1 One-to-One 建模](#21-one-to-one-建模)
  - [2.2 One-to-Many 建模](#22-one-to-many-建模)
  - [2.3 通用型 AiO 模型](#23-通用型-aio-模型)
- [3. 数据集](#3-数据集)
- [4. 损失函数与评价指标](#4-损失函数与评价指标)
- [5. Benchmark 快照](#5-benchmark-快照)
- [6. 未来方向](#6-未来方向)
- [7. 贡献指南](#7-贡献指南)
- [8. 引用](#8-引用)

---

# 1. 退化特定图像恢复

下文列出每个天气退化任务中被综述的深度学习方法。各任务表格按论文总览表中 "Category(架构类别)" 一列拆分;**Learning** 列与论文一致:SL(有监督)、UL(无监督)、SSL(半监督)、WSL(弱监督)。

## 1.1 图像去雾

*论文 §3.1 与 Table 2。图像去雾旨在从雾退化图像中恢复清晰场景。传统的(非深度学习)基于模型的方法依赖大气散射模型与手工先验(如暗通道先验 DCP);论文总览表收录深度学习方法。*

### CNN 方法

| Method | Venue | Learning | Network structure |
|---|---|---|---|
| DehazeNet (Cai et al., 2016) | TIP 2016 | SL | 4-Stage CNN with Maxout + BReLU |
| MSCNN (Ren et al., 2016) | ECCV 2016 | SL | Multi-scale Residual CNN |
| AOD-Net (Li et al., 2017) | ICCV 2017 | SL | Residual CNN + Multi-Scale |
| DCPDN (Zhang & Patel, 2018a) | CVPR 2018 | SL | Densely Connected Encoder-Decoder + U-Net + Pyramid Pooling |
| GFN (Ren et al., 2018) | CVPR 2018 | SL | Multi-Scale Residual Encoder-Decoder + Gated Fusion |
| GridDehazeNet (Liu et al., 2019) | ICCV 2019 | SL | Attention + Multi-scale CNN |
| FAMED-Net (Zhang & Tao, 2019) | TIP 2019 | SL | Multi-Scale CNN + Dense Connection |
| Deep DCP (Golts et al., 2019) | TIP 2019 | UL | Dilated Residual CNN |
| SSID (Li et al., 2019a) | TIP 2019 | SSL | Encoder-Decoder Architecture with Attention Mechanism |
| FFA-Net (Qin et al., 2020) | AAAI 2020 | SL | Residual + Attention CNN |
| MSCNN-HE (Ren et al., 2020) | IJCV 2020 | SL | Multi-scale + Coarse-scale + Fine-scale + Edge Guided |
| MSBDN (Dong et al., 2020a) | CVPR 2020 | SL | Multi-scale Residual U-Net + Boosted Decoder + DFF |
| CCDM (Zhang et al., 2020b) | CVPR 2020 | SSL | Cycle Reconstruction + Consistency Modules |
| PGC (Zhao et al., 2020) | TCSVT 2020 | SL | PGC-UNet |
| Light-DehazeNet (Ullah et al., 2021) | TIP 2021 | SL | Lightweight CNN + CVR Module |
| AECR-Net (Wu et al., 2021) | CVPR 2021 | SL | Autoencoder-style Residual CNN |
| Jo et al. (Jo & Sim, 2021) | CVPR 2021 | SL | Multi-Scale Residual CNN |
| PSD (Chen et al., 2021c) | CVPR 2021 | SSL | Backbone CNN + Physics-compatible Head + A-Net |
| Yu et al. (Yu et al., 2021b) | CVPR 2021 | SL | Two-Branch CNN + Attention + Ensemble Fusion |
| YOLY (Li et al., 2021) | IJCV 2021 | UL | Layer Disentanglement with J-Net, T-Net, A-Net |
| DCNet (Chen et al., 2021d) | ICME 2021 | SSL | Encoder-decoder + Depth/Transmission dual estimation |
| FSR (Liu et al., 2021c) | ACMMM 2021 | SSL | Hierarchical Encoder + Multi-scale Supervision |
| PTTD (Chen et al., 2024c) | ECCV 2024 | SL | Prompt Generation + Feature Adaptation + AECRNet |
| DCMPNet (Zhang et al., 2024c) | CVPR 2024 | SL | Dehazing branch + Depth estimation branch |
| Yin et al. (Yin & Liu, 2025) | Neural Netw. 2025 | UL | Multi-Branch + Multi-Scale Fusion |
| CoA (Ma et al., 2025) | CVPR 2025 | UL | Compressed student + bi-level domain adaptation |

### GAN 方法

| Method | Venue | Learning | Network structure |
|---|---|---|---|
| Dehaze-cGAN (Li et al., 2018b) | CVPR 2018 | SL | Encoder–Decoder Generator + CNN Discriminator |
| Cycle-Dehaze (Engin et al., 2018) | CVPR 2018 | UL | CycleGAN + Perceptual Encoder |
| DDN (Yang et al., 2018) | AAAI 2018 | UL | Disentangled Generator + Multi-Scale Discriminator |
| DehazeGAN (Zhu et al., 2018) | IJCAI 2018 | SL | Dense CNN Generator + PatchGAN Discriminator |
| HRGAN (Pang et al., 2018) | TCSVT 2018 | SL | UNTA + Discriminator |
| EPDN (Qu et al., 2019) | CVPR 2019 | SL | Multi-Resolution Generator + Multi-Scale Discriminator + Enhancer |
| CDNet (Dudhane & Murala, 2019) | WACV 2019 | UL | Encoder–Decoder Generator + CycleGAN Discriminator |
| Cycle-Defog2Refog (Liu et al., 2020) | TIP 2020 | UL | CycleGAN + Encoder-Decoder + Enhancer |
| Park et al. (Park et al., 2020) | TIP 2020 | UL + SL | CycleGAN + cGAN + Fusion CNN |
| FD-GAN (Dong et al., 2020b) | AAAI 2020 | SL | Densely Connected Encoder-Decoder + Fusion Discriminator |
| DW-GAN (Fu et al., 2021) | CVPR 2021 | SL | DWT Branch + Knowledge Adaptation Branch |
| TMS-GAN (Wang et al., 2021b) | TCSVT 2021 | SL | HgGAN + HrGAN |
| RefineDNet (Zhao et al., 2021) | TIP 2021 | WSL | RefineDNet |
| ODCR (Wang et al., 2024b) | CVPR 2024 | UL | Orthogonal MLPs + DWFC + PatchNCE |
| UBRFC-Net (Sun et al., 2024a) | Neural Netw. 2024 | UL | BCRF + Adaptive FCA |
| IPC-Dehaze (Fu et al., 2025) | CVPR 2025 | SL | Code-Predictor + Code-Critic + VQGAN |

### Transformer / Mamba 方法

| Method | Venue | Learning | Network structure |
|---|---|---|---|
| DeHamer (Guo et al., 2022) | CVPR 2022 | SL | Two-branch Transformer + Physical Prior Embedding |
| DehazeFormer (Song et al., 2023) | TIP 2023 | SL | Hierarchical Transformer + Local-Enhancement Modules |
| MB-TaylorFormer (Qiu et al., 2023) | ICCV 2023 | SL | Multi-Branch Taylor-Series Transformer |
| MoE-Mamba (Zhang et al., 2024a) | ACMMM 2024 | SL | Mamba Backbone + MoE Block + LMM-based Intensity-aware Block |
| DehazeXL (Chen et al., 2025a) | CVPR 2025 | SL | Patch encoder + global attention + decoder |

### 扩散模型方法

| Method | Venue | Learning | Network structure |
|---|---|---|---|
| DiffLI2D (Yang et al., 2024b) | ECCV 2024 | SL | Pretrained Diffusion Backbone + U-Net + H-space Decoder |
| Diff-Dehazer (Lan et al., 2025) | AAAI 2025 | UL | Stable Diffusion + CycleGAN + PAG + TAG |
| DiffDehaze (Wang et al., 2025b) | CVPR 2025 | SL | HazeGen + DiffDehaze + AccSamp |

## 1.2 图像去雨

*论文 §3.2 与 Table 3。图像去雨旨在去除雨线与雨致 veil 效应。论文把雨滴去除也归入本任务族(见下表的 RaindropAttention 与第 5 节 RainDrop benchmark)。*

### CNN 方法

| Method | Venue | Learning | Network structure |
|---|---|---|---|
| DerainNet (Fu et al., 2017a) | TIP 2017 | SL | Three Layers |
| DDN (Fu et al., 2017b) | CVPR 2017 | SL | Residual Network |
| JORDER (Yang et al., 2017) | CVPR 2017 | SL | Multi-Task Network |
| RESCAN (Li et al., 2018c) | ECCV 2018 | SL | Recurrent Network |
| PReNet (Ren et al., 2019) | CVPR 2019 | SL | Recursive Network |
| DAF-Net (Hu et al., 2019) | CVPR 2019 | SL | Residual Network |
| SPANet (Wang et al., 2019) | CVPR 2019 | SL | Multi-Stage Network |
| UMRL (Yasarla & Patel, 2019) | CVPR 2019 | SL | Multi-Scale Network |
| SSIR (Wei et al., 2019) | CVPR 2019 | SSL | Two Subnetworks |
| RaindropAttention (Quan et al., 2019) | ICCV 2019 | SL | U-Net Like Network |
| LPNet (Fu et al., 2019) | TNNLS 2019 | SL | Pyramid Network |
| JORDER-E (Yang et al., 2019) | TPAMI 2019 | SL | Multi-Task Network |
| MSPFN (Jiang et al., 2020) | CVPR 2020 | SL | Multi-Scale Network |
| DRD-Net (Deng et al., 2020) | CVPR 2020 | SL | Two Subnetworks |
| RCDNet (Wang et al., 2020) | CVPR 2020 | SL | Multi-Stage Network |
| Syn2Real (Yasarla et al., 2020) | CVPR 2020 | SSL | U-Net Like Network |
| MOSS (Huang et al., 2021) | CVPR 2021 | SSL | U-Net Like Network |
| SPDNet (Yi et al., 2021) | ICCV 2021 | SL | Recursive Network |
| UDRDR (Liu et al., 2021b) | ICCV 2021 | SSL | Three Subnetworks |
| UDGNet (Yu et al., 2021a) | ACMMM 2021 | UL | Multi-Task Network |
| EfficientDeRain (Guo et al., 2021) | AAAI 2021 | SL | U-Net Like Network |
| ESDNet (Song et al., 2024) | IJCAI 2024 | SL | SNN |
| PADUM (Xiao & Xia, 2025) | Neural Netw. 2025 | SL | APGD + SFPM |
| DEMore-Net (Wang et al., 2025c) | Neural Netw. 2025 | SL | Depth Estimation + MOR + HNB |
| CSUD (Dong et al., 2025) | CVPR 2025 | SL | U-shaped Dense Net + Cross-scale Spatial Feature |

### GAN 方法

| Method | Venue | Learning | Network structure |
|---|---|---|---|
| DID-MDN (Zhang & Patel, 2018b) | CVPR 2018 | SL | Multi-Stream Network |
| AttentGAN (Qian et al., 2018) | CVPR 2018 | SL | Recurrent Network |
| RR-GAN (Zhu et al., 2019b) | AAAI 2019 | UL | Multi-Scale Network |
| UD-GAN (Jin et al., 2019) | ICIP 2019 | UL | Multi-Task Network |
| ID-CGAN (Zhang et al., 2019) | TCSVT 2019 | SL | Two Subnetworks |
| HRGAN (Li et al., 2019b) | CVPR 2019 | SL | Multi-Stage Network |
| DerainCycleGAN (Wei et al., 2021) | TIP 2021 | UL | CycleGAN |
| VRGNet (Wang et al., 2021a) | CVPR 2021 | SL | Four Subnetworks |
| JRGR (Ye et al., 2021) | CVPR 2021 | SL | CycleGAN |
| DCD-GAN (Chen et al., 2022c) | CVPR 2022 | UL | CycleGAN |
| NLCL (Ye et al., 2022b) | CVPR 2022 | UL | Two Subnetworks |

### Transformer / Mamba 方法

| Method | Venue | Learning | Network structure |
|---|---|---|---|
| ELF (Jiang et al., 2022) | ACMMM 2022 | SL | Two Subnetworks |
| IDT (Xiao et al., 2022) | TPAMI 2022 | SL | U-Net Like Network |
| HCT-FFN (Chen et al., 2023b) | AAAI 2023 | SL | Multi-Stage Network |
| DRSformer (Chen et al., 2023a) | CVPR 2023 | SL | U-Net Like Network |
| SmartAssign (Wang et al., 2023c) | CVPR 2023 | SL | Multi-Task Network |
| SCD-Former (Guo et al., 2023) | ICCV 2023 | SL | Transformer + Cross-Layer Attention |
| RLP (Zhang et al., 2023b) | ICCV 2023 | SL | RLP + RPIM + DM |
| NeRD-Rain (Chen et al., 2024b) | CVPR 2024 | SL | U-Net Like Network |
| FADformer (Gao et al., 2024a) | ECCV 2024 | SL | FAD + Transformer + Cross-scale interaction |
| FreqMamba (Zou et al., 2024) | ACMMM 2024 | SL | U-Net + Triple-branch FreqSSM Block |
| MS-DEMamba (Cheng et al., 2025) | ACMMM 2025 | SL | Dual-branch Mamba + Dynamic Adaptive Scanning Blocks |

## 1.3 图像去雪

*论文 §3.3 与 Table 4。图像去雪旨在去除雪花与雪致 veil 效应。论文只以作者-年份引注形式给出、未给出方法名的行,下表按论文原样引用。*

### CNN 方法

| Method | Venue | Learning | Network structure |
|---|---|---|---|
| DesnowNet (Liu et al., 2018) | TIP 2018 | SL | Residual + Multi-scale |
| JSTASR (Chen et al., 2020) | ECCV 2020 | SL | JSTASR |
| HDCW-Net (Chen et al., 2021b) | ICCV 2021 | SL | Residual |
| DDMSNet (Zhang et al., 2021) | TIP 2021 | SL | Residual + DDMSNet + Multi-scale |
| Ye et al. (2022a) | ACCV 2022 | SL | Asym. Encoder-Decoder + Multi-scale Pyramid |
| HCSD-Net (Zhang et al., 2023d) | ACMMM 2023 | SL | RGB Branch + Hue Branch + Multi-Scale Fusion Module |
| PEUNet (Guo et al., 2025) | TCSVT 2025 | SL | Prior-Enhanced Unfolding |
| SnowMaster (Lai et al., 2025) | CVPR 2025 | SSL | CNN + MLLM eval + Mean-teacher |

### GAN 方法

| Method | Venue | Learning | Network structure |
|---|---|---|---|
| Li et al. (2019d) | IEEE Access 2019 | SL | Snow-mask estimation + composition GAN |
| DesnowGAN (Jaw et al., 2020) | TCSVT 2020 | SL | SR + Refinement + WGAN-GP |

### Transformer 方法

| Method | Venue | Learning | Network structure |
|---|---|---|---|
| SmartAssign (Wang et al., 2023c) | CVPR 2023 | SL | Multi-task Network |

> 除上表所列深度学习方法外,论文叙述部分还讨论了基于模型的去雪(如 MGF,Zheng et al., 2013)与可逆神经网络方法 InvDSNet(Quan et al., 2023),后者通过双路径结构将图像分解为雪层与干净层(论文 §3.3)。

## 1.4 其他不良天气恢复任务

*论文 §3.4。除去雾/去雨/去雪外,论文还综述了其他天气诱发的退化。这些任务在文中以段落叙述为主(多为引注簇,而非逐一命名的方法),下表按论文的分组复刻。*

| 退化类型 | 论文引用 | 说明(来自论文 §3.4) |
|---|---|---|
| 沙尘暴恢复 | (Liu et al., 2022; Liu et al., 2021a; Si et al., 2023) | 抑制空气中粉尘颗粒的影响:低对比度、颜色失真、类霾的重 veil |
| 暴雨(heavy rain) | (Li et al., 2019b; Wen et al., 2024; Zhang et al., 2024b) | 雨线、积水雨滴与薄雾混合,退化复杂且不均匀,超出常规去雨范畴 |
| 夜间霾 | (Cong et al., 2024; Jin et al., 2023; Liu et al., 2023; Yan et al., 2020; Zhang et al., 2017; Zhang et al., 2020a) | 低可见度 + 人造光源 + 传感器噪声,常需联合建模光照与透射率 |
| 结霜场景 | — | 半透明或不透明结晶图案遮挡图像内容 |

论文还讨论了耦合退化,如 雨+低光(L2RIRNet, Lin et al., 2025a;MS-DEMamba, Cheng et al., 2025 见 1.2 表)与 雨&霾 混合(论文 §3.6)。

## 1.5 基于视频的不良天气恢复

*论文 §3.5。视频恢复需在恢复每帧空间细节的同时保持相邻帧间的时间一致性。*

| 方法 | 任务 | 思路(来自论文 §3.5) |
|---|---|---|
| MAP-Net (Xu et al., 2023) | 视频去雾 | 物理先验建模 + 多范围时间对齐(记忆式物理先验引导模块 + 多范围场景辐射恢复模块) |
| SemiVDN (Wu et al., 2024) | 视频去雪 | 半监督;分布驱动对比正则化 + 先验引导的时间解耦专家模块;引入 85 段真实雪景视频 |
| Wang et al. (2023b) | 视频去雨 | 无监督、事件相机辅助框架:非对称分离、跨模态融合与跨模态对比学习 |

---

# 2. All-in-One 图像恢复

*论文 §4。AiO 方法在统一架构中处理多种天气退化。论文依据推理行为将其分为 **One-to-One 方法**(对所有输入走固定推理路径)与 **One-to-Many 方法**(借助退化相关线索——prompt、分类器、路由权重、专家选择等——自适应地调整推理)。论文 Table 5 给出该分类法的总览,下表与之对应。*

## 2.1 One-to-One 建模

### 共享主干 (Shared Backbone)

| Method | Key idea | Venue |
|---|---|---:|
| TransWeather (Valanarasu et al., 2022) | Plain Transformer for joint weather restoration | CVPR 2022 |
| Restormer (Zamir et al., 2022a) | Efficient Transformer with shared restoration path | CVPR 2022 |
| Uformer (Wang et al., 2022) | U-shaped Transformer with multi-scale fusion | CVPR 2022 |
| FocalNet (Cui et al., 2023) | Focal modulation in a unified backbone | ICCV 2023 |
| AIRFormer (Gao et al., 2024b) | Frequency-aware Transformer for unified restoration | TCSVT 2024 |
| Histoformer (Sun et al., 2024b) | Histogram-guided attention in a shared graph | ECCV 2024 |
| ACL (Gu et al., 2025) | Mamba-based linear attention for unified restoration | CVPR 2025 |

### 结构化路径 (Structured Path)

| Method | Key idea | Venue |
|---|---|---:|
| All-in-One (Li et al., 2020) | Fixed multi-encoder architecture with NAS fusion | CVPR 2020 |
| MPRNet (Zamir et al., 2021) | Progressive multi-stage restoration with fixed graph | CVPR 2021 |

## 2.2 One-to-Many 建模

### 混合专家 (MoE)

| Method | Key idea | Venue |
|---|---|---:|
| Chen et al. (2022b) | Expert learning with contrastive regularization | CVPR 2022 |
| AWRCP (Ye et al., 2023) | Codebook-guided weather expert fusion | ICCV 2023 |
| MetaWeather (Kim et al., 2024) | Meta-initialized experts for weather adaptation | ECCV 2024 |
| LDR (Yang et al., 2024a) | Language-guided expert routing | CVPR 2024 |
| SLER-IR (Peng et al., 2026) | Spherical layer-wise expert routing | arXiv 2026 |

### Prompt 条件 (Prompt Conditioning)

| Method | Key idea | Venue |
|---|---|---:|
| PromptIR (Potlapalli et al., 2023) | Learnable prompts for shared restoration | NeurIPS 2023 |
| MPerceiver (Ai et al., 2024) | Multimodal prompts for diffusion restoration | CVPR 2024 |
| AST (Zhou et al., 2024) | Adaptive sparse attention as conditional guidance | CVPR 2024 |
| DFPIR (Tian et al., 2025) | Degradation-aware feature prompts | CVPR 2025 |
| Wang et al. (2025a) | Feature-difference instructions with adapters | CVPR 2025 |

### 条件感知路由 (Condition-Aware Routing)

| Method | Key idea | Venue |
|---|---|---:|
| AiOENet (Liu et al., 2024a) | Classification-guided encoder-decoder routing | TIV 2024 |
| AirNet (Li et al., 2022) | Contrastive degradation representation for guided restoration | CVPR 2022 |
| WGWS-Net (Zhu et al., 2023) | Weather-guided dual-branch attention | CVPR 2023 |
| OKNet (Cui et al., 2024) | Adaptive omni-kernel receptive field routing | AAAI 2024 |
| GridFormer (Wang et al., 2024a) | Degradation-aware local-global path routing | IJCV 2024 |
| UHD-Processor (Liu et al., 2025) | Degradation-aware latent and frequency routing | CVPR 2025 |
| ILAWR (Lu et al., 2025) | Dynamic mask routing for continuous weather | CVPR 2025 |
| JarvisIR (Lin et al., 2025b) | VLM-guided expert orchestration | CVPR 2025 |

### 扩散类方法 (Diffusion-Based)

| Method | Key idea | Venue |
|---|---|---:|
| WeatherDiffusion (Özdenizci & Legenstein, 2023) | Conditional DDPM with weather embeddings | TPAMI 2023 |
| Diff-Plugin (Liu et al., 2024b) | Plugin-enhanced diffusion restoration | CVPR 2024 |
| DTPM (Ye et al., 2024) | Texture priors for conditional diffusion | CVPR 2024 |
| SemiDDM-weather (Long et al., 2025) | Teacher-student diffusion learning | Neural Netw. 2025 |
| ResFlow (Qin et al., 2025) | Reversible conditional diffusion flow | CVPR 2025 |
| Defusion (Luo et al., 2025) | Visual-instructed diffusion restoration | CVPR 2025 |

## 2.3 通用型 AiO 模型

*论文 §4.3。除面向天气的方法外,论文还综述了通用型 AiO 模型——在统一、任务无关的框架内处理多种退化因素(噪声、模糊、低光、压缩伪影等)。*

- **MIRNet** (Zamir et al., 2020) 与 **MIRNetv2** (Zamir et al., 2022b) — 多尺度残差设计以增强特征学习。
- **DGUNet** (Mou et al., 2022) — 将优化过程展开为可学习的阶段。
- **NAFNet** (Chen et al., 2022a) — 简化恢复架构以提升效率。
- **BIDeN** (Han et al., 2022) — 基于分解的灵活恢复策略。
- **AMIRNet** (Zhang et al., 2023a) — 层次化退化先验。
- **SwinIR** (Liang et al., 2021) — 移位窗口建模局部-全局依赖。
- **Fourmer** (Zhou et al., 2023) — 4D 特征交互。
- **AutoDIR** (Jiang et al., 2024) — 用扩散先验推断并适应退化类型。
- **Perceive-IR** (Zhang et al., 2025a) — 主干无关的 AiO 框架,通过质量感知 prompt 学习联合建模退化类型与程度。
- **UniUIR** (Zhang et al., 2025b) — 将统一恢复视为 all-in-one 学习器(IEEE TIP;见论文 §4.3 末尾)。

---

# 3. 数据集

*论文 §5.1。下表复刻论文数据集总览(论文 Table 6)。Size 与 S(合成)/R(真实)/S&R 列遵循论文;Paired 表示至少有一个子集提供对应的干净参考图像,同时包含合成与真实数据的集合可能还包括无配对真实图像。*

## 3.1 去雾数据集

| Dataset | Size | Syn/Real | Paired | Key idea | Venue |
|---|---|---|:---:|---|---:|
| FRIDA (Tarel et al., 2010) | 90 | S | ✓ | Small-Scale Synthetic Dataset with Heterogeneous Fog Simulation | IVS 2010 |
| FRIDA2 (Tarel et al., 2012) | 330 | S | ✓ | Diverse Synthetic Fog Dataset with Precise Physical Control | ITSM 2012 |
| NYU-Depth V2 (Silberman et al., 2012) | 1449 | R | ✗ | Densely Annotated RGB-D Indoor Dataset with Support Relations | ECCV 2012 |
| D-HAZY (Ancuti et al., 2016) | 1472 | S | ✓ | Depth-Guided Synthetic Indoor Hazy Dataset | ICIP 2016 |
| RESIDE (Li et al., 2018a) | 86,000+ | S&R | ✓ | Comprehensive Synthetic & Real-World Benchmark | TIP 2018 |
| I-HAZE (Ancuti et al., 2018a) | 35 | R | ✓ | Indoor Real Paired Haze Benchmark | ACIVS 2018 |
| O-HAZE (Ancuti et al., 2018b) | 45 | R | ✓ | Outdoor Real Paired Haze Benchmark | CVPR 2018 |
| Foggy Cityscapes (Sakaridis et al., 2018) | 20,550 | S | ✓ | Synthetic Fog Dataset with Semantic Labels from Real Driving Scenes | IJCV 2018 |
| HazeCOCO (Zhu et al., 2019a) | 700,000+ | S | ✓ | Large-Scale Multi-Source Synthetic Dataset with Semantic Labels | TCYB 2019 |
| Dense-HAZE (Ancuti et al., 2019) | 33 | R | ✓ | Real Dense Haze Benchmark | ICIP 2019 |
| NH-HAZE (Ancuti et al., 2020) | 55 | R | ✓ | Non-Homogeneous Real Haze Benchmark | CVPR 2020 |
| 4KID (Zheng et al., 2021) | 10,000 | S | ✓ | UHD Synthetic Dehazing Benchmark for Real-Time 4K Restoration | CVPR 2021 |
| Haze4K (Liu et al., 2021c) | 3,000/1,000 | S | ✓ | Paired Synthetic Dataset for Semi-Supervised Dehazing | ACMMM 2021 |

## 3.2 去雨数据集

| Dataset | Size | Syn/Real | Paired | Key idea | Venue |
|---|---|---|:---:|---|---:|
| Rain12 (Li et al., 2016) | 0/12 | S | ✓ | Small-scale benchmark with Photoshop-simulated rain; widely used for early testing | CVPR 2016 |
| Rain100L (Yang et al., 2017) | 1,800/100 | S | ✓ | Light rain simulation over 100 images; tests on sparse streaks | CVPR 2017 |
| Rain100H (Yang et al., 2017) | 1,800/100 | S | ✓ | Heavy and complex rain patterns; challenges model robustness | CVPR 2017 |
| Rain200L (Yang et al., 2017) | 1,800/200 | S | ✓ | Augmented light rain dataset for improved diversity | CVPR 2017 |
| Rain200H (Yang et al., 2017) | 1,800/200 | S | ✓ | Augmented heavy rain dataset; evaluates performance under dense rain | CVPR 2017 |
| RainDrop (Qian et al., 2018) | 1119 | R | ✓ | Real image pairs with glass-adhered raindrops; enables learning of severe occlusion restoration | CVPR 2018 |
| DID-Data (Zhang & Patel, 2018b) | 12,000/1,200 | S | ✓ | For rain streak removal, categorized by rain density levels | CVPR 2018 |
| Rain800 (Zhang et al., 2019) | 700/100 | S | ✓ | Synthetic paired dataset with 700 training and 100 test images | TCSVT 2019 |
| SPA-Data (Wang et al., 2019) | 28,500/1,000 | R | ✓ | Real-rain pairs constructed from videos with temporal priors and human supervision | CVPR 2019 |
| MPID (Li et al., 2019c) | 3,961/419 | S&R | ✓ | Multi-type & task-driven benchmark for synthetic and real-world deraining | CVPR 2019 |
| RainCityscapes (Hu et al., 2019) | 9,432/1,188 | S | ✓ | Depth-aware synthetic rain + fog based on Cityscapes | CVPR 2019 |
| Outdoor-Rain (Li et al., 2019b) | 9,000/1,500 | S | ✓ | Simulated heavy rain with depth-aware streaks and veiling effects | CVPR 2019 |
| Rain13K (Jiang et al., 2020) | 13,712/4,298 | S | ✓ | Large-scale low-resolution dataset with diverse rain patterns; foundation for modern deraining | CVPR 2020 |
| GT-RAIN (Ba et al., 2022) | 26,124/2,100 | R | ✓ | Fully real paired dataset captured in outdoor conditions | ECCV 2022 |
| 4K-Rain13k (Chen et al., 2024a) | 12,500/500 | S | ✓ | First UHD dataset with realistic rain geometry; supports high-res model evaluation and training | arXiv 2024 |
| HQ-RAIN (Chen et al., 2025b) | 4,500/500 | S | ✓ | High-fidelity synthetic dataset with perceptually realistic rain patterns | TPAMI 2025 |

## 3.3 去雪数据集

| Dataset | Size | Syn/Real | Paired | Key idea | Venue |
|---|---|---|:---:|---|---:|
| Snow100K (Liu et al., 2018) | 100,000+ | S&R | ✓ | Large-scale synthetic dataset with multi-scale snow particle modeling | TIP 2018 |
| SRRS (Chen et al., 2020) | 15,000+ | S&R | ✓ | Synthetic snow with multi-scale masks and physical veiling modeling | ECCV 2020 |
| CSD (Chen et al., 2021b) | 10,000 | S | ✓ | All-in-one synthetic snow dataset with flakes, streaks, and veiling | ICCV 2021 |
| SnowCityscapes (Zhang et al., 2021) | 2,000/2,000 | S | ✓ | Synthetic snow applied to Cityscapes for diverse semantic urban scenes | TIP 2021 |
| SnowKITTI2012 (Zhang et al., 2021) | 1,500/1,000 | S | ✓ | Snow overlays on KITTI for depth-aware, geometry-sensitive evaluation | TIP 2021 |
| RealSnow10K (Lai et al., 2025) | 10,000 | R | ✗ | Large-scale real-world unpaired dataset for unsupervised desnowing | CVPR 2025 |

## 3.4 All-in-One / 多天气数据集

| Dataset | Size | Syn/Real | Paired | Key idea | Venue |
|---|---|---|:---:|---|---:|
| All-weather (Valanarasu et al., 2022) | 18,069 | S&R | ✓ | Evaluate multi-weather restoration methods by combining images with different weather degradations | CVPR 2022 |
| AIR40K (Gao et al., 2024b) | 40,000 | S&R | ✓ | Combined multi-weather benchmark with fully paired samples for unified adverse weather restoration | TCSVT 2024 |
| HAC (Wan et al., 2025) | 316,000 | S | ✓ | Hybrid-weather benchmark covering 31 combinations of five adverse factors | PR 2025 |
| WeatherStream (Zhang et al., 2023c) | 202,000 | R | ✓ | Automated time-multiplexed real-weather pairs for multi-weather single-image restoration | CVPR 2023 |
| WeatherBench (Guan et al., 2025) | 42,002 | R | ✓ | Real paired benchmark captured under controlled rain, snow, and haze conditions | ACMMM 2025 |

## 3.5 论文 benchmark 使用的训练/测试划分

*论文 Table 7 —— 论文 §5.4 实验所用六个任务的数据集汇总。训练/测试规模按原表报告。*

| Type | Dataset | Train | Test |
|---|---|---|---|
| Dehazing | ITS (Li et al., 2018a) | 13,990 | - |
| Dehazing | OTS (Li et al., 2018a) | 72,135 | - |
| Dehazing | SOTS-indoor (Li et al., 2018a) | - | 500 |
| Dehazing | SOTS-outdoor (Li et al., 2018a) | - | 500 |
| Dehazing | Dense-Haze (Ancuti et al., 2019) | 28 | 5 |
| Dehazing | NH-HAZE (Ancuti et al., 2020) | 50 | 5 |
| Deraining | Rain100L (Yang et al., 2017) | 1800 | 100 |
| Deraining | Rain100H (Yang et al., 2017) | 1800 | 100 |
| Deraining | Rain800 (Zhang et al., 2019) | 700 | 100 |
| Deraining | SPA-Data (Wang et al., 2019) | 28,500 | 1000 |
| Deraining | DID-Data (Zhang & Patel, 2018b) | 12,000 | 1200 |
| Desnowing | Snow100K (Liu et al., 2018) | 50,000 | 50,000 |
| Desnowing | CSD (Chen et al., 2021b) | 8000 | 2000 |
| Desnowing | SRRS (Chen et al., 2020) | 2500 | 1000 |
| Desnowing | SnowKITTI2012 (Zhang et al., 2021) | 1500 | 1000 |
| Deraining&Dehazing | Outdoor-Rain (Li et al., 2019b) | 9000 | 750 |
| RainDrop Removal | RainDrop (Qian et al., 2018) | 861 | 58 |
| Multi-weather Restoration | All-weather (Valanarasu et al., 2022) | 18,069 | - |
| Multi-weather Restoration | Snow100K-S (Liu et al., 2018) | - | 16,611 |
| Multi-weather Restoration | Snow100K-L (Liu et al., 2018) | - | 16,801 |
| Multi-weather Restoration | RainDrop (Qian et al., 2018) | - | 58 |
| Multi-weather Restoration | Outdoor-Rain (Li et al., 2019b) | - | 750 |

---

# 4. 损失函数与评价指标

*论文 §5.2 将损失分为单任务设定与 AiO 专用两类;论文 §5.3 将指标分为全参考 (FR) 与无参考 (NR) 两类。下表中的说明均引自论文。*

## 4.1 损失函数

### 单任务损失 (论文 §5.2.1)

| Loss | Purpose / notes (from Survey §5.2.1) |
|---|---|
| Pixel-wise loss | Directly supervises pixel-level fidelity; common variants include L1 (MAE), L2 (MSE), and Charbonnier losses. L1 is robust to outliers and sharp edges (suitable for deraining); L2 penalizes significant errors (often used in dehazing/desnowing); Charbonnier balances both and is widely used for stable convergence in CNN- and Transformer-based models |
| Edge / gradient loss | Preserves local structures and reduces over-smoothing by measuring differences in image derivatives — particularly beneficial for rain streaks and snowflakes, which tend to be blurred without gradient supervision |
| SSIM loss | Improves structural fidelity by comparing luminance, contrast, and structure; often combined with L1 or perceptual losses; MS-SSIM variant further improves delicate and global structures |
| Perceptual loss | Compares deep features from a pretrained network (e.g., VGG-16) rather than raw pixels; mitigates over-smoothing and improves high-frequency detail; extensions include feature matching and LPIPS |
| Adversarial loss | Trains a discriminator to distinguish real from restored images (e.g., PatchGAN for local textures); usually combined with pixel-wise or perceptual losses for training stability and high-frequency details |

### 多任务 / AiO 损失 (论文 §5.2.2)

| Loss | Purpose / notes (from Survey §5.2.2) |
|---|---|
| Task-discriminative loss | Crucial for expert-based or routing architectures (e.g., AWRCP); prevents the collapse of routing networks or condition embeddings that causes feature entanglement. Methods such as Chen et al. (2022b) and LDR employ gating regularization or contrastive learning; SLER-IR adds a hyperspherical contrastive loss for angular separability of degradation representations |
| Progressive guidance loss | Applies supervision (e.g., L1 or SSIM) to intermediate outputs of each stage/branch in multi-stage or multi-branch models (e.g., MPRNet, AiOENet); promotes coarse-to-fine detail recovery and stabilizes gradient flow |
| Condition-aware auxiliary loss | Auxiliary objectives that improve the discriminability and stability of degradation representations (feature alignment, condition consistency, or task separation); typically combined with standard reconstruction losses |

## 4.2 评价指标

### 全参考 (FR) 指标

| Metric | Notes (from Survey §5.3.1) |
|---|---|
| PSNR | Peak Signal-to-Noise Ratio; pixel-wise fidelity based on the mean squared error |
| SSIM | Structural Similarity Index; luminance, contrast, and structure — better aligned with human perception |
| LPIPS | Learned Perceptual Image Patch Similarity; deep features of pre-trained networks (VGG, AlexNet) for perceptual differences that low-level pixel metrics often miss |
| FSIM | Feature Similarity Index; phase congruency and gradient magnitude for structural consistency |
| WPSNR, CIEDE2000, VSI, VIF, UQI | Specialized FR metrics (weighted PSNR; color difference; visual saliency-based index; visual information fidelity; universal quality index) appearing in specific benchmarks |

### 无参考 (NR) 指标

| Metric | Notes (from Survey §5.3.2) |
|---|---|
| NIQE | Naturalness Image Quality Evaluator; NSS-based model deviations, without a reference |
| BRISQUE | Blind/Referenceless Image Spatial Quality Evaluator; spatial-domain NSS, robust for mild distortions |
| PIQE / SSEQ | Perception-based Image Quality Evaluator; Spatial-Spectral Entropy-based Quality |
| MUSIQ | Multi-scale Image Quality; transformer-based holistic quality prediction aligned with human MOS |
| CLIP-IQA | Uses multi-modal CLIP embeddings, often measuring consistency between the restored image and textual prompts (e.g., "a clear image") — relevant for evaluating prompt-based AiO models |
| PI / Ma | Perceptual Index (e.g., combining NIQE and Ma); aggregated perceptual quality from super-resolution, now used in adverse-weather benchmarks |
| MetaIQA / NIMA | Predict human preference scores to supplement traditional NR measures |

> **分布级与下游指标。** 论文还提及分布级指标(如 FID)与下游任务指标(检测的 mAP、语义分割的 IoU),用于评估恢复是否提升实际视觉感知(论文 §5.3)。

---

# 5. Benchmark 快照

*论文 §5.4 与 Tables 8–13。以下数值全部复刻自论文。如论文 §5.4.1 所述,结果为"compiled from the corresponding publications and publicly reported results rather than reproduced under a unified implementation"(来自各原文与公开报告结果,而非统一实现下的复现),解读时需考虑各方法训练设置与评测协议的差异。"-" 表示论文未报告。论文表中横线以上为单任务方法、以下为 AiO 模型,为便于阅读我们分成两个区块;分析详见论文 §5.4.2。*

## 5.1 图像去雾(论文 Table 8:SOTS-indoor, SOTS-outdoor, Dense-Haze, NH-HAZE)

### 单任务方法

| Method | SOTS-indoor PSNR↑ / SSIM↑ | SOTS-outdoor PSNR↑ / SSIM↑ | Dense-Haze PSNR↑ / SSIM↑ | NH-HAZE PSNR↑ / SSIM↑ |
|---|---|---:|---:|---:|
| DehazeNet (Cai et al., 2016) | 19.82 / 0.821 | 24.75 / 0.927 | 13.84 / 0.430 | 16.62 / 0.520 |
| AOD-Net (Li et al., 2017) | 20.51 / 0.816 | 24.14 / 0.920 | 13.14 / 0.410 | 15.40 / 0.570 |
| GridDehazeNet (Liu et al., 2019) | 32.16 / 0.984 | 30.86 / 0.982 | 13.31 / 0.370 | 13.80 / 0.540 |
| FFA-Net (Qin et al., 2020) | 36.39 / 0.989 | 33.57 / 0.984 | 14.39 / 0.450 | 19.87 / 0.690 |
| MSBDN (Dong et al., 2020a) | 33.67 / 0.985 | 33.48 / 0.982 | 15.37 / 0.490 | 19.23 / 0.710 |
| AECR-Net (Wu et al., 2021) | 37.17 / 0.990 | - | 15.80 / 0.466 | 19.88 / 0.625 |
| DeHamer (Guo et al., 2022) | 36.63 / 0.988 | 35.18 / 0.986 | 16.62 / 0.560 | 20.66 / 0.684 |
| DehazeFormer (Song et al., 2023) | 40.05 / 0.996 | 34.95 / 0.984 | 16.29 / 0.510 | - |
| MB-TaylorFormer (Qiu et al., 2023) | 42.63 / 0.994 | 38.09 / 0.991 | 16.44 / 0.566 | 19.43 / 0.638 |

### AiO 方法

| Method | SOTS-indoor PSNR↑ / SSIM↑ | SOTS-outdoor PSNR↑ / SSIM↑ | Dense-Haze PSNR↑ / SSIM↑ | NH-HAZE PSNR↑ / SSIM↑ |
|---|---|---:|---:|---:|
| Restormer (Zamir et al., 2022a) | 38.88 / 0.991 | - | 15.78 / 0.550 | - |
| FocalNet (Cui et al., 2023) | 40.82 / 0.996 | 37.71 / 0.995 | 17.07 / 0.630 | 20.43 / 0.790 |
| OKNet (Cui et al., 2024) | 40.79 / 0.996 | 37.68 / 0.995 | 16.92 / 0.640 | 20.48 / 0.800 |

## 5.2 图像去雨(论文 Table 9:Rain100L, Rain100H, Rain800, SPA-Data, DID-Data)

### 单任务方法

| Method | Rain100L PSNR↑ / SSIM↑ | Rain100H PSNR↑ / SSIM↑ | Rain800 PSNR↑ / SSIM↑ | SPA-Data PSNR↑ / SSIM↑ | DID-Data PSNR↑ / SSIM↑ |
|---|---|---:|---:|---:|---:|
| DerainNet (Fu et al., 2017a) | 27.03 / 0.884 | 14.92 / 0.592 | 22.77 / 0.810 | 30.78 / 0.912 | - |
| DDN (Fu et al., 2017b) | 32.38 / 0.926 | 24.64 / 0.849 | 21.16 / 0.732 | 36.16 / 0.946 | 30.97 / 0.912 |
| DID-MDN (Zhang & Patel, 2018b) | 23.79 / 0.773 | 17.35 / 0.524 | 21.89 / 0.795 | 34.68 / 0.930 | 29.66 / - |
| RESCAN (Li et al., 2018c) | 29.80 / 0.881 | 29.62 / 0.872 | 24.09 / 0.841 | 38.11 / 0.980 | 33.38 / 0.942 |
| PReNet (Ren et al., 2019) | 32.44 / 0.950 | 30.11 / 0.905 | 26.61 / 0.902 | 40.16 / 0.982 | 33.17 / 0.948 |
| MSPFN (Jiang et al., 2020) | 33.50 / 0.948 | 30.63 / 0.898 | 27.50 / 0.876 | 43.43 / 0.984 | 33.72 / 0.955 |
| IDT (Xiao et al., 2022) | 34.74 / 0.958 | 31.10 / 0.914 | 27.84 / 0.915 | 47.34 / 0.993 | 34.89 / 0.962 |
| DRSformer (Chen et al., 2023a) | 35.23 / 0.959 | 31.17 / 0.913 | 28.35 / 0.919 | 48.54 / 0.992 | 35.35 / 0.965 |
| FADformer (Gao et al., 2024a) | 35.80 / 0.960 | 31.48 / 0.916 | 28.42 / 0.920 | 49.21 / 0.993 | 35.48 / 0.966 |

### AiO 方法

| Method | Rain100L PSNR↑ / SSIM↑ | Rain100H PSNR↑ / SSIM↑ | Rain800 PSNR↑ / SSIM↑ | SPA-Data PSNR↑ / SSIM↑ | DID-Data PSNR↑ / SSIM↑ |
|---|---|---:|---:|---:|---:|
| MPRNet (Zamir et al., 2021) | 34.95 / 0.959 | 28.53 / 0.872 | 28.09 / 0.891 | 45.00 / 0.99 | 33.99 / 0.959 |
| Restormer (Zamir et al., 2022a) | 37.57 / 0.974 | 29.46 / 0.889 | 29.12 / 0.908 | 46.25 / 0.991 | 35.29 / 0.964 |
| PromptIR (Potlapalli et al., 2023) | 37.04 / 0.979 | 28.69 / 0.877 | 29.07 / 0.887 | - | - |

## 5.3 图像去雪(论文 Table 10:Snow100K, CSD, SRRS, SnowKITTI2012)

### 单任务方法

| Method | Snow100K PSNR↑ / SSIM↑ | CSD PSNR↑ / SSIM↑ | SRRS PSNR↑ / SSIM↑ | SnowKITTI2012 PSNR↑ / SSIM↑ |
|---|---|---:|---:|---:|
| DesnowNet (Liu et al., 2018) | 30.50 / 0.940 | 20.13 / 0.810 | 20.38 / 0.840 | 30.12 / 0.899 |
| JSTASR (Chen et al., 2020) | 23.12 / 0.860 | 27.96 / 0.880 | 25.82 / 0.890 | 29.45 / 0.881 |
| HDCW-Net (Chen et al., 2021b) | 31.54 / 0.950 | 29.06 / 0.910 | 27.78 / 0.920 | 35.57 / 0.965 |
| DDMSNet (Zhang et al., 2021) | 30.76 / 0.910 | 28.79 / 0.900 | 27.03 / 0.910 | 36.86 / 0.977 |
| PEUNet (Guo et al., 2025) | 34.11 / 0.959 | 38.27 / 0.990 | - | 38.53 / 0.988 |

### AiO 方法

| Method | Snow100K PSNR↑ / SSIM↑ | CSD PSNR↑ / SSIM↑ | SRRS PSNR↑ / SSIM↑ | SnowKITTI2012 PSNR↑ / SSIM↑ |
|---|---|---:|---:|---:|
| All-in-One (Li et al., 2020) | 26.07 / 0.880 | 26.31 / 0.870 | 24.98 / 0.880 | - |
| MPRNet (Zamir et al., 2021) | 33.87 / 0.950 | 33.98 / 0.970 | 30.37 / 0.960 | 35.71 / 0.970 |
| TransWeather (Valanarasu et al., 2022) | 31.82 / 0.930 | 31.76 / 0.930 | 28.29 / 0.920 | 31.57 / 0.940 |
| Restormer (Zamir et al., 2022a) | 34.67 / 0.950 | 35.43 / 0.970 | 32.24 / 0.960 | 36.42 / 0.980 |
| Chen et al. (2022b) | 34.37 / 0.950 | 33.89 / 0.960 | 30.82 / 0.960 | 34.17 / 0.960 |
| FocalNet (Cui et al., 2023) | 33.53 / 0.950 | 37.18 / 0.990 | 31.34 / 0.980 | - |
| OKNet (Cui et al., 2024) | 33.75 / 0.950 | 37.99 / 0.990 | 31.70 / 0.980 | 38.14 / 0.986 |

## 5.4 去雨 & 去雾(论文 Table 11:Outdoor-Rain)

| Method | Outdoor-Rain PSNR↑ / SSIM↑ |
|---|---:|
| HRGAN (Li et al., 2019b) | 21.56 / 0.855 |
| MPRNet (Zamir et al., 2021) | 28.03 / 0.919 |
| Restormer (Zamir et al., 2022a) | 29.97 / 0.921 |
| AIRFormer (Gao et al., 2024b) | 29.21 / 0.935 |
| RainHazeDiff64 (Özdenizci & Legenstein, 2023) | 28.38 / 0.932 |
| RainHazeDiff128 (Özdenizci & Legenstein, 2023) | 26.84 / 0.915 |
| GridFormer (Wang et al., 2024a) | 28.49 / 0.921 |
| DTPM (Ye et al., 2024) | 30.99 / 0.934 |
| ResFlow (Qin et al., 2025) | 32.82 / 0.936 |

## 5.5 RainDrop 去除(论文 Table 12:RainDrop)

| Method | RainDrop PSNR↑ / SSIM↑ |
|---|---:|
| AttentGAN (Qian et al., 2018) | 31.59 / 0.917 |
| RaindropAttention (Quan et al., 2019) | 31.44 / 0.926 |
| CCN (Quan et al., 2021) | 31.34 / 0.929 |
| IDT (Xiao et al., 2022) | 31.87 / 0.931 |
| AirNet (Li et al., 2022) | 31.32 / 0.925 |
| PromptIR (Potlapalli et al., 2023) | 32.03 / 0.938 |
| RainDropDiff64 (Özdenizci & Legenstein, 2023) | 32.29 / 0.942 |
| RainDropDiff128 (Özdenizci & Legenstein, 2023) | 32.43 / 0.933 |
| GridFormer (Wang et al., 2024a) | 32.92 / 0.940 |
| DTPM (Ye et al., 2024) | 32.72 / 0.944 |
| Wang et al. (2025a) | 33.70 / 0.939 |
| Defusion (Luo et al., 2025) | 33.81 / 0.967 |

## 5.6 跨数据集多天气恢复(论文 Table 13)

*所有 AiO 模型仅在 All-weather 数据集上训练,直接应用到各 benchmark 数据集。*

| Method | Snow100K-S PSNR↑ / SSIM↑ | Snow100K-L PSNR↑ / SSIM↑ | RainDrop PSNR↑ / SSIM↑ | Outdoor-Rain PSNR↑ / SSIM↑ | Params | MACs |
|---|---|---|---|---|---|---|
| All-in-One (Li et al., 2020) | 31.78 / 0.921 | 28.33 / 0.882 | 31.12 / 0.927 | 24.71 / 0.898 | 44.00M | 12.26G |
| TransWeather (Valanarasu et al., 2022) | 32.51 / 0.934 | 29.31 / 0.888 | 30.17 / 0.916 | 28.83 / 0.900 | 21.90M | 5.64G |
| Restormer (Zamir et al., 2022a) | 36.08 / 0.959 | 30.28 / 0.912 | 30.91 / 0.928 | 30.21 / 0.921 | 26.10M | 140.99G |
| Chen et al. (2022b) | 34.80 / 0.948 | 30.24 / 0.902 | 30.99 / 0.927 | 29.92 / 0.917 | 28.71M | 24.56G |
| WeatherDiff64 (Özdenizci & Legenstein, 2023) | 35.83 / 0.957 | 30.09 / 0.904 | 30.71 / 0.931 | 29.64 / 0.931 | 82.92M | 475.16G |
| WeatherDiff128 (Özdenizci & Legenstein, 2023) | 35.02 / 0.952 | 29.58 / 0.894 | 29.66 / 0.923 | 29.72 / 0.922 | 85.56M | 263.45G |
| AWRCP (Ye et al., 2023) | 36.92 / 0.965 | 31.92 / 0.934 | 31.93 / 0.931 | 31.39 / 0.933 | - | - |
| GridFormer (Wang et al., 2024a) | 37.46 / 0.964 | 31.71 / 0.923 | 32.39 / 0.936 | 31.87 / 0.934 | 30.12M | 251.35G |
| MPerceiver (Ai et al., 2024) | 36.23 / 0.957 | 31.02 / 0.916 | 33.21 / 0.929 | 31.25 / 0.925 | - | - |
| DTPM (Ye et al., 2024) | 37.01 / 0.966 | 30.92 / 0.917 | 32.72 / 0.944 | 30.99 / 0.934 | - | - |
| Histoformer (Sun et al., 2024b) | 37.41 / 0.966 | 32.16 / 0.926 | 33.06 / 0.944 | 32.08 / 0.939 | 16.61M | 91.56G |

---

# 6. 未来方向

*论文 §6 提出不良天气图像恢复的五个未来方向,以下引号内为该方向的原文开头句。*

1. **弥合合成-真实差距 (Bridging the synthetic-to-real gap)。** *"The synthetic-to-real gap remains one of the most critical bottlenecks in adverse weather restoration."* 近期研究走向测试时自适应、半监督学习、域自适应与真实配对数据。

2. **迈向实用且自适应的 all-in-one 恢复。** *"Although AiO restoration provides a promising framework for handling unknown and mixed degradations, current models still face challenges in task interference, controllable adaptation, and deployment complexity."* 动态路由、prompt 条件调制与专家特化被重点强调,以 JarvisIR 为代表的 agentic/多专家编排框架亦被提及。

3. **建立任务效用中心的评测体系 (Task-utility-centric benchmarking)。** *"Traditional full-reference metrics such as PSNR and SSIM remain useful, but they are insufficient for evaluating restoration quality in practical adverse weather scenarios."* 论文主张评估任务效用——下游检测、分割与跟踪性能——以及混合天气、未知天气与时间一致性等设定。

4. **提升效率与可部署性。** *"As restoration models become more powerful, their computational and memory costs also increase."* 轻量架构、动态推理、模型压缩与加速扩散采样,是自动驾驶、移动平台与边缘设备上的可行方向。

5. **构建可信的开世界恢复系统。** *"Real-world adverse weather restoration is inherently an open-world problem."* 开世界自适应、持续学习与多模态恢复(雷达、LiDAR、热成像、事件相机等互补传感器)前景广阔;领域应从 benchmark 驱动走向长期真实部署中依然可靠的可信系统。

---

# 7. 贡献指南

欢迎提交论文建议、纠错、数据集链接与 benchmark 更新。请提交 issue 或 pull request,并遵循 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

**范围说明:** 论文中已有的条目必须与论文的分组、venue/年份及描述一致;论文之外的新工作欢迎补充,但需明确标注为社区新增。Benchmark 数值以论文中的汇总为准(§5.4.1);社区新增条目需注明来源与评测设置。

# 8. 引用

如果您觉得本综述或本仓库对您的研究有帮助,请引用:

```bibtex
@article{song2027deep,
  title   = {Deep image restoration in adverse weather: A survey},
  author  = {Song, Zhenbo and Li, Ruixin and Zhang, Zhenyuan and Wang, Tao and Lu, Jianfeng and Yu, Xin and Zhang, Kaihao},
  journal = {Neural Networks},
  volume  = {205},
  pages   = {109472},
  year    = {2027},
  doi     = {10.1016/j.neunet.2026.109472}
}
```

## 致谢

仓库版式参考了 [TaoWangzj/Awesome-Image-Restoration](https://github.com/TaoWangzj/Awesome-Image-Restoration)。其中的分类法、方法组织、数据集汇总、benchmark 数值与图片均复刻自我们以 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 开放获取发表的综述论文 *Deep Image Restoration in Adverse Weather: A Survey*。
