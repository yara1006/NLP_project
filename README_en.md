# adv-fraud-nlp

[![Tests](https://img.shields.io/badge/tests-8%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

🌐 **[English](README_en.md)** | **[中文](README.md)**

Adversarial attack and robustness evaluation for Chinese fraud text classification. Adapts [PromptAttack](https://github.com/microsoft/promptbench) (ICLR 2024) with LLM-generated adversarial paraphrases, SQLite API caching, and RoBERTa fine-tuning.

> Based on: *An LLM can Fool Itself: A Prompt-Based Adversarial Attack* (ICLR 2024)

---

## Overview

This project builds a complete pipeline for evaluating the robustness of Chinese fraud telephone text classifiers against LLM-generated adversarial attacks:

```
Raw CSV Data → Data Conversion → RoBERTa Training → Adversarial Generation → Robustness Evaluation
```

**Key results:**
- RoBERTa original accuracy: ~92%
- Adversarial accuracy (after LLM attack): ~65%
- Attack Success Rate (ASR): ~29%
- BERTScore similarity: >0.94

---

## Project Structure

```
adv-fraud-nlp/
├── src/                          # Core library
│   ├── __init__.py
│   ├── Call.py                   # DashScope/Qwen API client + SQLite cache
│   ├── PromptAttack.py           # Chinese-adapted PromptAttack (lang="zh")
│   ├── Predict.py                # LLM output label parser
│   └── Dataset.py                # HuggingFace Dataset wrapper
├── scripts/                      # Executable scripts
│   ├── train_bert_classifier.py  # RoBERTa fine-tuning
│   ├── bert_classifier_predictor.py  # Batch prediction
│   ├── convert_data_format.py    # CSV format conversion
│   ├── generate_adv_sentences.py # Adversarial sample generation
│   ├── eval_adv_llm.py           # LLM/RoBERTa dual-mode evaluation
│   ├── eval_adv_dialogue.py      # Dialogue-level evaluation
│   ├── eval_single_dialogue_file.py  # Single-file evaluation
│   └── robustness_eval.py        # Original PromptAttack entry
── tests/
│   └── test_data.py              # Data conversion + cache tests
├── docs/
│   ├── ARCHITECTURE.md           # System architecture & pipeline
│   ├── RETROSPECTIVE.md          # Design decisions & lessons learned
│   └── CLAUDE.md                 # Project rules for AI coding
├── data/
│   ├── original_data/            # Raw fraud call CSV data
│   └── *.json                    # PromptAttack GLUE benchmark data
├── pyproject.toml                # Project config & dependencies
├── requirements.txt              # Legacy pip requirements
└── LICENSE                       # MIT License
```

---

## Quick Start

### 1. Install

```bash
# Clone and install
git clone https://github.com/yara1006/adv-fraud-nlp.git
cd adv-fraud-nlp
pip install -e ".[dev]"
```

### 2. Convert Data Format

```bash
python scripts/convert_data_format.py
```

Converts raw fraud call CSV (`specific_dialogue_content`, `is_fraud`) to standard `text/label` format.

### 3. Train RoBERTa Classifier

```bash
python scripts/train_bert_classifier.py
```

Fine-tunes `hfl/chinese-roberta-wwm-ext` for binary fraud classification. Model saved to `roberta_classifier/`.

### 4. Generate Adversarial Samples

```bash
export DASHSCOPE_API_KEY="your-api-key"
python scripts/generate_adv_sentences.py
```

Uses Qwen/DashScope to generate Chinese adversarial paraphrases for fraud samples.

### 5. Evaluate Robustness

```bash
# LLM classifier mode
python scripts/eval_adv_llm.py

# RoBERTa classifier mode
CLS_TYPE=roberta python scripts/eval_adv_llm.py

# Dialogue-level evaluation
python scripts/eval_adv_dialogue.py --step original --cls_type roberta
python scripts/eval_adv_dialogue.py --step adversarial --cls_type roberta
python scripts/eval_adv_dialogue.py --step calculate --attack_type asymmetric
```

### 6. Run Tests

```bash
python -m pytest tests/ -v
```

---

## Core Concepts

### Adversarial Attack via LLM Rewriting

Instead of word-level perturbations (synonym replacement, character swap), this project uses LLM-generated paraphrases that:
- Preserve core fraud intent
- Change surface form to evade classifiers
- Maintain semantic similarity (BERTScore >0.94)

### Chinese PromptAttack Adaptation

Original PromptAttack uses English GLUE tasks. This project adapts it for:
- Chinese tokenization (`jieba` instead of NLTK)
- Custom Chinese perturbation instructions
- Fraud classification labels: `["非诈骗", "诈骗"]`

### SQLite API Cache

`Call.py` implements prompt-response caching to:
- Reduce redundant API calls during experiment iteration
- Support thread-safe parallel generation
- Track cache hit rates for cost monitoring

---

## Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Original Accuracy | Correct / Total | Model accuracy on non-attacked samples |
| Adversarial Accuracy | Correct / Total | Model accuracy on adversarial samples |
| ASR | (Orig - Adv) / Orig | Attack Success Rate |
| BERTScore | F1 similarity | Semantic preservation between original and adversarial |

---

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)** — System pipeline, module descriptions, design decisions
- **[Retrospective](docs/RETROSPECTIVE.md)** — Lessons learned, trade-offs, quantitative metrics
- **[Project Rules](docs/CLAUDE.md)** — Coding standards, test requirements, environment setup

---

## Requirements

- Python 3.9+
- PyTorch 2.0+ (GPU recommended for training)
- DashScope API key (for adversarial generation)

Install dependencies:

```bash
pip install -e ".[dev]"
```

Or use legacy requirements:

```bash
pip install -r requirements.txt
```

---

## License

[MIT License](LICENSE)

---

## Citation

If you use this project in your research, please cite the original PromptAttack paper:

```bibtex
@inproceedings{zhang2024llm,
  title={An LLM can Fool Itself: A Prompt-Based Adversarial Attack},
  author={Zhang, Zhuosheng and others},
  booktitle={ICLR},
  year={2024}
}
```
