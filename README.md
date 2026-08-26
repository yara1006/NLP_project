# adv-fraud-nlp

[![Tests](https://img.shields.io/badge/tests-8%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

🌐 **[English](README_en.md)** | **[中文](README.md)**

中文诈骗文本对抗攻击与鲁棒性评估系统。基于 [PromptAttack](https://github.com/microsoft/promptbench) (ICLR 2024) 扩展，使用 LLM 生成对抗样本、SQLite API 缓存和 RoBERTa 微调。

> 基于论文：*An LLM can Fool Itself: A Prompt-Based Adversarial Attack* (ICLR 2024)

---

## 项目概述

本项目构建了完整的中文诈骗电话文本分类器鲁棒性评估流程，针对 LLM 生成的对抗攻击进行评估：

```
原始 CSV 数据 → 数据转换 → RoBERTa 训练 → 对抗样本生成 → 鲁棒性评估
```

**关键结果：**
- RoBERTa 原始准确率：~92%
- 对抗样本准确率（LLM 攻击后）：~65%
- 攻击成功率（ASR）：~29%
- BERTScore 相似度：>0.94

---

## 项目结构

```
adv-fraud-nlp/
├── src/                          # 核心库
│   ├── __init__.py
│   ├── Call.py                   # DashScope/Qwen API 客户端 + SQLite 缓存
│   ├── PromptAttack.py           # 中文适配的 PromptAttack (lang="zh")
│   ├── Predict.py                # LLM 输出标签解析器
│   └── Dataset.py                # HuggingFace Dataset 封装
├── scripts/                      # 可执行脚本
│   ├── train_bert_classifier.py  # RoBERTa 微调
│   ├── bert_classifier_predictor.py  # 批量预测
│   ├── convert_data_format.py    # CSV 格式转换
│   ├── generate_adv_sentences.py # 对抗样本生成
│   ├── eval_adv_llm.py           # LLM/RoBERTa 双模式评估
│   ├── eval_adv_dialogue.py      # 对话级评估
│   ├── eval_single_dialogue_file.py  # 单文件评估
│   └── robustness_eval.py        # 原始 PromptAttack 入口
├── tests/
│   └── test_data.py              # 数据转换 + 缓存测试
├── docs/
│   ├── ARCHITECTURE.md           # 系统架构与流程
│   ├── RETROSPECTIVE.md          # 设计决策与经验教训
│   └── CLAUDE.md                 # AI 编程协作项目规则
├── data/
│   ├── original_data/            # 原始诈骗电话 CSV 数据
│   └── *.json                    # PromptAttack GLUE 基准数据
├── pyproject.toml                # 项目配置与依赖
├── requirements.txt              # 传统 pip 依赖
└── LICENSE                       # MIT 许可证
```

---

## 快速开始

### 1. 安装

```bash
# 克隆并安装
git clone https://github.com/yara1006/adv-fraud-nlp.git
cd adv-fraud-nlp
pip install -e ".[dev]"
```

### 2. 转换数据格式

```bash
python scripts/convert_data_format.py
```

将原始诈骗电话 CSV（`specific_dialogue_content`, `is_fraud`）转换为标准 `text/label` 格式。

### 3. 训练 RoBERTa 分类器

```bash
python scripts/train_bert_classifier.py
```

微调 `hfl/chinese-roberta-wwm-ext` 进行二分类诈骗检测。模型保存到 `roberta_classifier/`。

### 4. 生成对抗样本

```bash
export DASHSCOPE_API_KEY="your-api-key"
python scripts/generate_adv_sentences.py
```

使用 Qwen/DashScope 为诈骗样本生成中文对抗改写文本。

### 5. 评估鲁棒性

```bash
# LLM 分类器模式
python scripts/eval_adv_llm.py

# RoBERTa 分类器模式
CLS_TYPE=roberta python scripts/eval_adv_llm.py

# 对话级评估
python scripts/eval_adv_dialogue.py --step original --cls_type roberta
python scripts/eval_adv_dialogue.py --step adversarial --cls_type roberta
python scripts/eval_adv_dialogue.py --step calculate --attack_type asymmetric
```

### 6. 运行测试

```bash
python -m pytest tests/ -v
```

---

## 核心概念

### 基于 LLM 改写的对抗攻击

不同于词级别扰动（同义词替换、字符交换），本项目使用 LLM 生成的改写文本：
- 保留核心诈骗意图
- 改变表面形式以逃避分类器
- 保持语义相似度（BERTScore >0.94）

### 中文 PromptAttack 适配

原始 PromptAttack 使用英文 GLUE 任务。本项目适配：
- 中文分词（使用 `jieba` 替代 NLTK）
- 自定义中文扰动指令
- 诈骗分类标签：`["非诈骗", "诈骗"]`

### SQLite API 缓存

`Call.py` 实现 prompt-response 缓存：
- 减少实验迭代中的重复 API 调用
- 支持线程安全的并行生成
- 追踪缓存命中率以监控成本

---

## 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| 原始准确率 | 正确数 / 总数 | 模型在未攻击样本上的准确率 |
| 对抗准确率 | 正确数 / 总数 | 模型在对抗样本上的准确率 |
| ASR | (原始 - 对抗) / 原始 | 攻击成功率 |
| BERTScore | F1 相似度 | 原始与对抗样本间的语义保持度 |

---

## 文档

- **[架构说明](docs/ARCHITECTURE.md)** — 系统流程、模块说明、设计决策
- **[复盘文档](docs/RETROSPECTIVE.md)** — 经验教训、权衡取舍、量化指标
- **[项目规则](docs/CLAUDE.md)** — 编码规范、测试要求、环境配置

---

## 依赖要求

- Python 3.9+
- PyTorch 2.0+（训练推荐 GPU）
- DashScope API key（对抗样本生成）

安装依赖：

```bash
pip install -e ".[dev]"
```

或使用传统依赖：

```bash
pip install -r requirements.txt
```

---

## 许可证

[MIT 许可证](LICENSE)

---

## 引用

如果在研究中使用本项目，请引用原始 PromptAttack 论文：

```bibtex
@inproceedings{zhang2024llm,
  title={An LLM can Fool Itself: A Prompt-Based Adversarial Attack},
  author={Zhang, Zhuosheng and others},
  booktitle={ICLR},
  year={2024}
}
```
