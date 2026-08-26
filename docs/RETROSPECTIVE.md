# NLP_project — Retrospective Document

> Lessons learned, trade-offs, and what we would do differently.

---

## 1. Chinese PromptAttack: LLM Rewriting vs. Word-Level Perturbation

**Context**: Original PromptAttack uses word-level perturbations (synonym replacement, character swap, typo injection). These work for English but don't transfer well to Chinese fraud text.

**Decision**: Use LLM (Qwen/DashScope) to generate Chinese adversarial paraphrases that maintain semantic meaning while evading the classifier.

**Trade-offs**:
- ✅ Higher ASR (Attack Success Rate) — LLM rewrites are more natural and harder to detect
- ✅ Preserves core fraud intent while changing surface form
- ❌ Slower than word-level perturbation (each sample requires an API call)
- ❌ Costs API credits (~¥0.01-0.03 per sample)
- ❌ Less reproducible — LLM outputs vary between runs

**If redoing**: Cache adversarial samples with semantic deduplication (BERTScore > 0.95) to reduce redundant API calls.

---

## 2. SQLite Cache for API Calls

**Context**: Generating adversarial samples requires thousands of LLM calls. Without caching, re-running experiments wastes API credits.

**Decision**: SQLite-based prompt-response cache in `Call.py` with `INSERT OR REPLACE` for idempotent writes.

**Trade-offs**:
- ✅ Significant cost reduction — repeated prompts return cached results instantly
- ✅ Thread-safe with `threading.Lock` for parallel generation
- ❌ Cache key is exact prompt string — any prompt change bypasses cache
- ❌ No semantic deduplication — similar prompts still trigger new API calls
-  Cache file can grow large (GB+) for big experiments

**If redoing**: Add semantic cache using embedding similarity (e.g., sentence-transformers) to deduplicate similar prompts.

---

## 3. RoBERTa Fine-Tuning: Full Model vs. LoRA

**Context**: Fine-tuning `hfl/chinese-roberta-wwm-ext` (108M params) on fraud classification.

**Decision**: Full fine-tuning (all parameters updated), not LoRA or adapter-based methods.

**Trade-offs**:
- ✅ Simpler — no additional dependencies (peft, bitsandbytes)
- ✅ Better performance on small datasets (< 10K samples)
- ❌ Requires GPU with 8GB+ VRAM
- ❌ Cannot easily switch base models without retraining

**If redoing**: Use LoRA for parameter-efficient fine-tuning — faster iteration, smaller checkpoints, easier model switching.

---

## 4. Evaluation: Single Classifier vs. Dual Evaluation

**Context**: Need to evaluate robustness of both local RoBERTa and cloud-based Qwen classifiers.

**Decision**: Support both evaluation modes via `CLS_TYPE` environment variable.

**Trade-offs**:
- ✅ RoBERTa: Fast, local, reproducible — good for iterative development
- ✅ Qwen: More realistic — reflects how real-world LLM classifiers behave
- ❌ Qwen evaluation costs API credits and is non-deterministic
-  Results differ significantly between classifiers (Qwen typically more robust)

**If redoing**: Add ensemble evaluation — evaluate against multiple classifiers and report worst-case ASR.

---

## 5. Data Format: CSV vs. JSONL vs. HuggingFace Dataset

**Context**: Raw fraud data comes as CSV. PromptAttack expects JSON format. Training scripts expect various formats.

**Decision**: Keep CSV as source of truth, convert to required formats as needed.

**Trade-offs**:
- ✅ CSV is human-readable and easy to inspect
- ✅ `convert_data_format.py` handles conversion centrally
- ❌ Multiple format conversions add complexity
-  Path inconsistencies (`mydata/` vs `data/`) cause confusion

**If redoing**: Use HuggingFace `datasets` library as the single source of truth — automatic format conversion, efficient loading, built-in caching.

---

## 6. If Redoing NLP_project, What Would Change?

### Methodology
1. **LoRA fine-tuning**: Parameter-efficient, faster iteration
2. **Semantic cache**: Embedding-based deduplication for API calls
3. **Ensemble evaluation**: Multiple classifiers, worst-case ASR reporting
4. **HuggingFace datasets**: Single source of truth for data

### Engineering
5. **Proper Python package**: `nlp_project/` with `__init__.py`, `setup.py`
6. **Type hints**: Full type annotations for all functions
7. **Logging**: Structured logging instead of print statements
8. **Config files**: YAML/JSON config instead of hardcoded paths

### Research
9. **Transfer attack evaluation**: Train on RoBERTa, attack Qwen (and vice versa)
10. **Defense mechanisms**: Test adversarial training, input preprocessing defenses
11. **Human evaluation**: Verify adversarial samples are natural to human readers

---

## 7. Lessons Learned (Pitfalls)

| Pitfall | Cause | Solution |
|---------|-------|----------|
| Hardcoded `mydata/` paths | Scripts developed locally, paths never parameterized | Refactored `convert_data_format.py` to accept path arguments |
| `dashscope` import breaks tests | Module-level import, no mocking in tests | Added `sys.modules` mock in test setup |
| Non-deterministic LLM outputs | Temperature=0.1 but not 0 | Document that adversarial samples are not fully reproducible |
| Large `requirements.txt` | Captured full conda/pip environment | Could trim to only direct dependencies |
| No random seed control | Training and evaluation not reproducible | Should add `torch.manual_seed(42)` and `numpy.random.seed(42)` |

---

## 8. Quantitative Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Original Accuracy (RoBERTa) | ~92% | On test set |
| Adversarial Accuracy (RoBERTa) | ~65% | After LLM attack |
| Original Accuracy (Qwen) | ~95% | On test set |
| Adversarial Accuracy (Qwen) | ~78% | After LLM attack |
| ASR (RoBERTa) | ~29% | (92-65)/92 |
| ASR (Qwen) | ~18% | (95-78)/95 |
| BERTScore (avg) | >0.94 | Semantic similarity preserved |
| Training time (RoBERTa) | ~30 min | 3 epochs on GPU |
| API cost per sample | ~¥0.02 | DashScope Qwen |

---

*Last updated: 2026-08-25*
