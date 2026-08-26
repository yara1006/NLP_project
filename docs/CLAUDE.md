# CLAUDE.md — NLP_project Rules

## WHAT

NLP_project is a research project for Chinese fraud telephone text adversarial attack and robustness evaluation. It adapts the PromptAttack method (ICLR 2024) for Chinese fraud detection, using LLM-generated adversarial paraphrases to evaluate classifier robustness.

Tech stack: Python 3.9+, PyTorch, HuggingFace Transformers, DashScope/Qwen API, jieba, pandas, scikit-learn, pytest.

## WHY — Immutable Constraints

- **Reproducibility**: Set random seeds (`torch.manual_seed(42)`, `numpy.random.seed(42)`) for all experiments.
- **API key security**: Never commit DashScope API keys to Git. Use environment variable `DASHSCOPE_API_KEY`.
- **Data privacy**: Fraud call data is sensitive. Do not share raw data externally.
- **Cost control**: Use SQLite cache to avoid redundant API calls. Monitor API spending.

## HOW — Install, Run, Test, Debug

### Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run
```bash
# 1. Convert data format
python convert_data_format.py

# 2. Train RoBERTa classifier
python train_bert_classifier.py

# 3. Generate adversarial samples (requires DASHSCOPE_API_KEY)
export DASHSCOPE_API_KEY="your-key"
python generate_adv_sentences.py

# 4. Evaluate robustness
python eval_adv_llm.py              # LLM mode
CLS_TYPE=roberta python eval_adv_llm.py  # RoBERTa mode
```

### Test
```bash
python -m pytest tests/ -v
```

### Debug
- API call logs: `api_debug.log`
- SQLite cache: `log.db` (inspect with `sqlite3 log.db`)
- Training logs: stdout + TensorBoard (if configured)

## TEST — What Must Be Verified When Changing Code

- After changing `convert_data_format.py`: MUST run `python -m pytest tests/ -v`
- After changing `Call.py` cache logic: MUST verify cache hit/miss behavior
- After changing `PromptAttack.py`: MUST verify Chinese label mapping (`["非诈骗", "诈骗"]`)
- After changing `train_bert_classifier.py`: MUST verify model saves to `roberta_classifier/`
- External dependencies (DashScope) are mocked in tests — tests do NOT call real APIs

## Project Structure Quick Reference

```
NLP_project/
├── Call.py                         # DashScope/Qwen API client + SQLite cache
├── PromptAttack.py                 # Chinese-adapted PromptAttack (lang="zh")
├── Predict.py                      # LLM output label parser
├── Dataset.py                      # HuggingFace Dataset wrapper
├── train_bert_classifier.py        # RoBERTa fine-tuning
├── bert_classifier_predictor.py    # Local classifier batch prediction
── convert_data_format.py          # CSV format conversion (parameterized)
├── generate_adv_sentences.py       # Adversarial sample generation
├── eval_adv_llm.py                 # LLM/RoBERTa dual-mode evaluation
├── eval_adv_dialogue.py            # Dialogue-level evaluation
├── robustness_eval.py              # Original PromptAttack entry
├── tests/
│   └── test_data.py                # Data conversion + cache tests
└── data/
    ├── original_data/              # Raw fraud call CSV
    ── *.json                      # PromptAttack GLUE data
```

## Coding Standards

- Python 3.9+ compatibility
- All public functions MUST have docstrings (Google style)
- New modules go in project root (no package structure yet)
- New tests go in `tests/test_*.py`
- Do NOT commit API keys, large data files, or model checkpoints
- Use `logging` module instead of `print()` for production code
