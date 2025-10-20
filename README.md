<div>

# 【AAAI 2024 🔥】CLIP 在文本描述行人检索中的经验研究
[![Paper](http://img.shields.io/badge/Paper-AAAI_2024-Green.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/27801)
[![Paper](http://img.shields.io/badge/Paper-arxiv.2308.10045-FF6B6B.svg)](https://arxiv.org/abs/2308.10045)
</div>

本仓库提供 TBPS-CLIP（Text-based Person Search）的官方 PyTorch 实现，对应论文《An Empirical Study of CLIP for Text-based Person Search》。

如果你对文本描述行人检索方向感兴趣，欢迎同时阅读我们的小伙伴项目：
- 【ACM MM 2023】[Text-based Person Search without Parallel Image-Text Data](https://arxiv.org/abs/2305.12964)
- 【IJCAI 2023】[RaSa: Relation and Sensitivity Aware Representation Learning for Text-based Person Search](https://arxiv.org/abs/2305.13653)
- 【ICASSP 2022】[Learning Semantic-Aligned Feature Representation for Text-based Person Search](https://arxiv.org/abs/2112.06714)

## 说明
更多实验细节与拓展结果已附在 [arXiv 版本](https://arxiv.org/abs/2308.10045)的附录中。

## 项目概览
我们重新审视了 [CLIP](https://arxiv.org/abs/2103.00020) 在数据增广和损失函数设计中的关键策略，提出了面向文本描述行人检索的强大基线 TBPS-CLIP。

<img src="image/intro.png" width="550">

## 环境准备
- 默认实验环境：4 张 NVIDIA A40（48GB）GPU，CUDA 11.7。
- 推荐使用 Python 3.9+，并通过 `requirements.txt` 安装依赖：

```sh
pip install -r requirements.txt
```

## 数据与预训练模型下载
1. 数据集：
   - 从 [链接](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) 下载 CUHK-PEDES。
   - 从 [链接](https://github.com/zifyloo/SSAN) 下载 ICFG-PEDES。
   - 从 [链接](https://github.com/NjtechCVLab/RSTPReid-Dataset) 下载 RSTPReid。
2. 标注文件：在 [Google Drive](https://drive.google.com/file/d/1C5bgGCABtuzZMaa2n4Sc0qclUvZ-mqG9/view?usp=drive_link) 中获取整理好的 JSON 标注。
3. 预训练权重：
   - 原版 CLIP：下载 OpenAI 发布的 [ViT-B/16 权重](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)。
   - AltCLIP：使用 Hugging Face Hub 中的 `BAAI/AltCLIP` 系列模型，详细说明见下文。

## 配置文件
在 `config/config.yaml` 与 `config/s.config.yaml` 中填写：
- `data.annotation_file`：数据标注文件路径。
- `data.image_root`：图像目录。
- `model.clip_pretrained` 或 `model.altclip_pretrained`：对应模型权重路径（AltCLIP 支持本地缓存目录）。
- 其余超参（学习率、batch size 等）可根据设备资源调整。

## 训练与评估
使用 `torchrun` 启动分布式训练：

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 \
 torchrun --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 \
 main.py
```

若想使用简化配置，可加上 `--simplified` 选项：

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 \
 torchrun --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=4 \
 main.py --simplified
```

## AltCLIP 多语言扩展
AltCLIP 模型现已与 TBPS-CLIP 完整集成，可提供中文、英文及多种语言描述的统一编码能力。本节从差异说明、依赖配置、训练改动和调试脚本四个方面给出详细指南。

### 与原版 CLIP 的核心差异
- **文本编码器升级为 XLM-R。** AltCLIP 采用 XLM-RoBERTa 作为文本骨干，在 `model/altclip_adapter.py` 中由适配器对齐到原 `CLIP` 类的接口，原有损失函数、优化逻辑可无缝复用。
- **表征维度提升至 768。** AltCLIP 默认输出 768 维特征，需将配置中的 `model.embed_dim`、`model.projection_dim` 等字段同步改为 768；若使用自定义头部，请确保线性层输入尺寸一致。
- **分词器改为 Hugging Face 实现。** `text_utils/tokenizer.py` 中新增 `AltCLIPTokenizer`，基于 `transformers.AutoTokenizer` 调用 `BAAI/AltCLIP` 的词表。启用 AltCLIP 时会自动屏蔽原本依赖 BPE 的数据增强选项。
- **预处理与图像归一化略有差异。** AltCLIP 使用自带的 `AltCLIPProcessor` 完成图像 resize、中心裁剪与归一化；适配器会在 `build_altclip_clip` 中返回 `processor`，方便在自定义脚本中共享同一套预处理流程。
- **权重来源为 Hugging Face Hub。** 默认联网状态下首次运行会自动下载；若在离线环境，请先用 `transformers-cli download BAAI/AltCLIP` 或脚本缓存相关权重，并在配置中指定本地目录。

### 安装与配置步骤
1. 安装额外依赖：
   ```sh
   pip install transformers>=4.35.0 sentencepiece safetensors
   ```
2. 在 `config/config.yaml` 中启用 AltCLIP：
   ```yaml
   model:
     backbone: altclip          # 新增字段，取值 clip 或 altclip
     embed_dim: 768
     tokenizer_type: altclip
     altclip_pretrained: "BAAI/AltCLIP"
   experiment:
     use_altclip_processor: true
     mlm: false                 # AltCLIP 不支持原有 BPE-MLM 增强
   ```
3. 如果需要自定义图像大小或分辨率，请同步修改 `data.image_size`，并确认 `AltCLIPProcessor` 的 resize 插值策略满足需求。

### 快速体验脚本
- 运行 `misc/quick_start_altclip.py` 可以在几分钟内完成一次中英文描述的正负样本前向测试：
  ```sh
  python misc/quick_start_altclip.py --device cuda:0 --text "红色上衣的女生" "a man wearing a blue coat"
  ```
- 脚本会打印图文相似度矩阵与 logit_scale、logit_bias 等关键参数，验证 AltCLIP 是否加载成功。
- 如需复现论文训练流程，可将脚本中的 `build_altclip_clip` 导入自己的数据管线，只需替换模型构建部分即可。

### 在训练流水线中切换至 AltCLIP
1. 调整 `options.py` 中的命令行参数或 YAML 配置，使 `--backbone altclip` 被解析。
2. 在 `main.py` 中调用 `build_altclip_clip`，接收 `clip_model, tokenizer, processor` 三元组。
3. 数据加载阶段使用 `processor(images=..., return_tensors="pt")` 处理图像，并将文本批次传入新的 tokenizer。
4. 训练/评估脚本其余部分保持不变；若使用自定义头部，需要根据 768 维特征重新初始化线性层。

### 离线部署建议
- 使用以下命令提前下载权重及分词器：
  ```sh
  python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('BAAI/AltCLIP', cache_dir='./hf_cache'); AutoTokenizer.from_pretrained('BAAI/AltCLIP', cache_dir='./hf_cache')"
  ```
- 将配置中的 `altclip_pretrained`、`tokenizer_cache_dir` 指向 `./hf_cache`，即可在无网络环境中复现训练与推理。

## 模型权重
| 模型 | CUHK-PEDES | ICFG-PEDES | RSTPReid |
|:--:|:--:|:--:|:--:|
| **TBPS-CLIP (ViT-B/16)** | [下载](https://drive.google.com/file/d/1m_3pKanUWHQHeJ-zt-QeRXs7bmay-U5P/view?usp=drive_link) | [下载](https://drive.google.com/file/d/1az4z5b_ADXR7DcysPB5giOl52LjWDCSu/view?usp=drive_link) | [下载](https://drive.google.com/file/d/1qMUAsH-1lzkWUFQsUvUKTY0J6ZuGkYd6/view?usp=drive_link) |
| **简化版 TBPS-CLIP (ViT-B/16)** | [下载](https://drive.google.com/file/d/1W5oFZK9WNHMfy0OOaYQBzPsP1LZR80bT/view?usp=drive_link) | [下载](https://drive.google.com/file/d/1UoLd-MQ8tYJ7YPgCbh3nVSVYnJ9a_TG5/view?usp=drive_link) | [下载](https://drive.google.com/file/d/18zlc3q3Sze5rx3TqcfEeZEjrQXUTpcQF/view?usp=drive_link) |

## 鸣谢
- [CLIP](https://arxiv.org/abs/2103.00020) —— TBPS-CLIP 的核心架构来源。

## 引用
如果本仓库对你的研究或项目有所帮助，欢迎 star 🌟 并引用 📑 我们的论文：

```
@inproceedings{cao2024empirical,
  title={An Empirical Study of CLIP for Text-Based Person Search},
  author={Cao, Min and Bai, Yang and Zeng, Ziyin and Ye, Mang and Zhang, Min},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={465--473},
  year={2024}
}
```

## 许可证
本项目基于 MIT License 开源，具体条款见仓库根目录的 `LICENSE` 文件。
