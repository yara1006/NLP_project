# NLP_project — Architecture Document

> Chinese fraud text adversarial attack and robustness evaluation system.

## 1. System Overview

This project implements a complete pipeline for adversarial attack research on Chinese fraud telephone text classification:

1. **Data Processing**: Convert raw fraud call CSV data to standard text/label format
2. **Classifier Training**: Fine-tune Chinese RoBERTa for fraud detection (binary classification)
3. **Adversarial Attack**: Adapt PromptAttack for Chinese text using LLM-generated paraphrases
4. **Robustness Evaluation**: Compare model performance on original vs. adversarial samples

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA (CSV)                             │
│  specific_dialogue_content | is_fraud | fraud_type | ...     │
└────────────────────┬────────────────────────────────────────┘
                     │ convert_data_format.py
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 STANDARD DATA (text/label)                    │
│  text: "你好，我是客服..." | label: 0 (非诈骗) / 1 (诈骗)     │
└────────────────────┬────────────────────────────────────────┘
                     │
         ───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐    ┌─────────────────────────┐
│ RoBERTa Trainer │    │  PromptAttack (Chinese)  │
│  (local model)  │    │  (LLM adversarial gen)   │
└────────┬────────┘    └────────────┬────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐    ┌─────────────────────────┐
│  roberta_classifier/  │  adv_samples/           │
│  (saved weights)  │    │  (adversarial texts)   │
└────────────────┘    └────────────┬────────────┘
         │                          │
         ───────────┬──────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  ROBUSTNESS EVALUATION                        │
│  ┌──────────────────┐  ┌──────────────────────────────┐    │
│  │ Original Accuracy │  │ Adversarial Accuracy         │    │
│  │ (RoBERTa or Qwen) │  │ (RoBERTa or Qwen)            │    │
│  └──────────────────┘  └──────────────────────────────┘    │
│                          ASR = (Orig_Acc - Adv_Acc) / Orig_Acc │
└─────────────────────────────────────────────────────────────┘
```

## 3. Core Modules

### 3.1 Data Processing (`convert_data_format.py`)
- Converts raw fraud call CSV to standard `text/label` format
- `convert_csv(input, output)` — parameterized for testability
- `create_small_test_file()` — creates 5-row subset for quick testing

### 3.2 LLM Call & Cache (`Call.py`)
- `LLMLogSql` — SQLite-based prompt-response cache (reduces API costs)
- `LLMCall` — DashScope/Qwen API wrapper with retry logic
- Thread-safe caching with `threading.Lock`

### 3.3 Chinese PromptAttack (`PromptAttack.py`)
- Extends original PromptAttack with `lang="zh"` support
- Uses `jieba` for Chinese tokenization instead of NLTK
- Custom Chinese perturbation instructions for fraud text
- Label mapping: `["非诈骗", "诈骗"]` for `mydata` dataset

### 3.4 Classifier Training (`train_bert_classifier.py`)
- Fine-tunes `hfl/chinese-roberta-wwm-ext` for binary classification
- Default: max_length=160, batch_size=16, epochs=3
- Saves tokenizer and model to `roberta_classifier/`

### 3.5 Adversarial Sample Generation (`generate_adv_sentences.py`)
- Reads dialogue sentence data, generates adversarial rewrites for fraud samples
- Non-fraud samples kept as-is
- Supports checkpoint resume (continues from processed line count)

### 3.6 Evaluation Scripts
| Script | Purpose |
|--------|---------|
| `eval_adv_llm.py` | Compare original vs. adversarial on Qwen/RoBERTa |
| `eval_adv_dialogue.py` | Dialogue-level evaluation (original, unilateral, bilateral attack) |
| `eval_single_dialogue_file.py` | Single-file LLM classification evaluation |
| `robustness_eval.py` | Original PromptAttack robustness evaluation entry |

## 4. Key Design Decisions

### 4.1 SQLite Cache for API Calls
**Why**: Adversarial sample generation requires thousands of LLM calls; caching prevents redundant API costs during experiment iteration.

**Trade-off**: Cache key is the full prompt string — any prompt change bypasses cache. No semantic deduplication.

### 4.2 Chinese Perturbation via LLM Rewriting
**Why**: Traditional word-level perturbations (synonym replacement, character swap) don't transfer well to Chinese fraud detection. LLM rewriting produces more natural and effective adversarial samples.

**Trade-off**: LLM rewriting is slower and costs API credits, but produces higher ASR (Attack Success Rate).

### 4.3 Label Mapping for Chinese Task
**Why**: Original PromptAttack uses English GLUE labels. Chinese fraud task uses `["非诈骗", "诈骗"]`.

**Implementation**: `PromptAttack.__init__` checks `self.dataset == "mydata"` and sets appropriate labels.

## 5. Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| `Original Accuracy` | Correct / Total | Model accuracy on original (non-attacked) samples |
| `Adversarial Accuracy` | Correct / Total | Model accuracy on adversarial samples |
| `ASR` | (Orig_Acc - Adv_Acc) / Orig_Acc | Attack Success Rate — proportion of correctly-classified originals that become misclassified |
| `BERTScore` | F1 between original and adversarial | Semantic similarity — ensures adversarial samples preserve meaning |

## 6. File Structure

```
NLP_project/
├── Call.py                         # DashScope/Qwen API client + SQLite cache
├── PromptAttack.py                 # Chinese-adapted PromptAttack logic
├── Predict.py                      # LLM output label parser
── Dataset.py                      # HuggingFace Dataset wrapper
├── train_bert_classifier.py        # RoBERTa fine-tuning script
├── bert_classifier_predictor.py    # Local classifier batch prediction
├── convert_data_format.py          # CSV format conversion (parameterized)
── generate_adv_sentences.py       # Adversarial sample generation
├── eval_adv.py                     # Adversarial evaluation
├── eval_adv_llm.py                 # LLM/RoBERTa dual-mode evaluation
├── eval_adv_dialogue.py            # Dialogue-level evaluation
├── eval_single_dialogue_file.py    # Single-file evaluation
├── robustness_eval.py              # Original PromptAttack entry point
├── requirements.txt                # Python dependencies
├── tests/
│   └── test_data.py                # Data conversion + cache tests
── data/
    ├── original_data/              # Raw fraud call data
    ── *.json                      # PromptAttack GLUE data
```

---

**Last updated**: 2026-08-25
