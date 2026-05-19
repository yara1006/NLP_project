# NLP_project：中文诈骗文本对抗攻击与鲁棒性评估

本项目基于 PromptAttack 思路，面向中文诈骗通话文本分类任务，构建了从数据格式转换、中文文本对抗样本生成、分类模型训练，到鲁棒性评估的完整实验流程。项目同时支持本地 RoBERTa/BERT 分类器与通义千问 DashScope API 分类器，用于比较原始样本和对抗样本下模型的识别能力变化。

## 项目目标

- 针对中文诈骗通话文本构建二分类任务：`非诈骗` / `诈骗`。
- 微调中文 RoBERTa/BERT 分类模型，作为本地诈骗文本检测器。
- 改造 PromptAttack，使其支持中文文本、中文分词和中文攻击提示。
- 调用 Qwen/DashScope 生成对抗样本，模拟诈骗话术被改写后的检测绕过场景。
- 评估原始文本与对抗文本上的准确率、攻击成功率（ASR）等指标。

## 项目结构

```text
NLP_project/
├── Call.py                         # DashScope/Qwen API 调用与 SQLite 缓存
├── PromptAttack.py                 # PromptAttack 主逻辑，已加入中文任务适配
├── Predict.py                      # 基于 LLM 输出的标签解析预测器
├── Dataset.py                      # HuggingFace Dataset 包装类
├── train_bert_classifier.py        # 中文 RoBERTa/BERT 诈骗分类器训练脚本
├── bert_classifier_predictor.py    # 本地分类器批量预测封装
├── convert_data_format.py          # 原始 CSV 转 text/label 标准格式
├── generate_adv_sentences.py       # 逐句生成中文对抗样本
├── eval_adv.py                     # 对抗样本评估脚本
├── eval_adv_llm.py                 # LLM/RoBERTa 双模式对抗评估
├── eval_adv_dialogue.py            # 对话级原始/对抗/双边对抗评估
├── eval_single_dialogue_file.py    # 单个对抗文件的 LLM 分类评估
├── robustness_eval.py              # 原 PromptAttack 鲁棒性评估入口
├── requirements.txt                # Python 依赖
└── data/
    ├── original_data/              # 中文诈骗通话原始数据
    │   ├── trainResult.csv
    │   ├── testResult.csv
    │   └── test_small.csv
    ├── sst-2.json                  # PromptAttack GLUE 对抗数据
    ├── qnli.json
    ├── qqp.json
    ├── rte.json
    ├── mnli-m.json
    └── mnli-mm.json
```

## 核心模块说明

### 1. 数据处理

`convert_data_format.py` 将原始诈骗通话 CSV 转换为统一的 `text,label` 格式：

- 输入字段：`specific_dialogue_content`、`is_fraud`
- 输出字段：`text`、`label`
- 标签映射：`False/0 -> 非诈骗`，`True/1 -> 诈骗`

当前仓库数据位于 `data/original_data/`，部分脚本历史路径写作 `mydata/...`。如果直接运行相关脚本，需要根据实际目录调整路径，或将 `data/original_data` 复制/软链接为脚本期望的 `mydata/original_data`。

### 2. 本地分类器训练

`train_bert_classifier.py` 使用 HuggingFace Transformers 微调中文文本分类模型：

- 默认预训练模型：`hfl/chinese-roberta-wwm-ext`
- 最大长度：`160`
- Batch size：`16`
- Epochs：`3`
- 输出目录：`roberta_classifier/`

训练完成后会保存 tokenizer 和模型权重，可供 `bert_classifier_predictor.py`、`eval_adv_llm.py` 等评估脚本加载。

### 3. 中文 PromptAttack 改造

`PromptAttack.py` 在原始 PromptAttack 的基础上加入中文适配：

- 支持 `lang="zh"` 参数切换中文攻击流程。
- 使用 `jieba` 进行中文分词，替代英文 NLTK 分词。
- 为 `mydata` 自定义数据集增加 `非诈骗` / `诈骗` 标签映射。
- 构造中文攻击目标和攻击指导，要求模型在保持核心语义的同时生成更容易误导分类器的改写文本。
- 支持词级修改比例、语义相似度、分类器预测结果等约束，用于筛选有效对抗样本。

### 4. LLM 调用与缓存

`Call.py` 封装 DashScope/Qwen 调用逻辑，并使用 SQLite 记录 prompt-response 缓存：

- 减少重复 API 调用成本。
- 支持失败重试和原始响应日志记录。
- 默认使用 DashScope `Generation.call` 接口。

建议通过环境变量传入密钥：

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

不要将真实 API Key 写入代码或提交到仓库。

### 5. 对抗样本生成

`generate_adv_sentences.py` 读取逐句对话数据，对诈骗样本生成对抗改写文本：

- 输入：`mydata/convert_Data/{left|right}_dialogue_sentences.csv`
- 输出：`mydata/new_{left|right}_adv/{side}_dialogue_sentences_adv_sample_strategy_0.csv`
- 仅对诈骗样本执行攻击；非诈骗样本保留原文。
- 支持断点续跑：如果输出文件已存在，会从已处理行数继续。

### 6. 鲁棒性评估

项目提供多个评估入口：

- `eval_adv_llm.py`：比较原始样本和对抗样本在 Qwen 或 RoBERTa 分类器上的表现。
- `eval_adv_dialogue.py`：按对话级别组装左右对话，支持原始、单边对抗、双边对抗三种评估模式。
- `eval_single_dialogue_file.py`：对单个对抗文件进行 LLM 分类评估。

核心指标：

- `Original Accuracy`：原始样本分类准确率。
- `Adversarial Accuracy`：对抗样本分类准确率。
- `ASR`：Attack Success Rate，原始预测正确但对抗样本预测错误的比例。

## 环境准备

建议使用 Python 3.9+，并创建独立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果在 Windows PowerShell 中使用：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

项目依赖包含 PyTorch、Transformers、Datasets、DashScope、BERTScore、jieba、pandas、scikit-learn 等。若安装 GPU 版本 PyTorch，请根据本机 CUDA 版本参考 PyTorch 官方安装命令。

## 快速运行

### 1. 转换数据格式

如需按脚本默认路径运行，可先准备 `mydata/testResult.csv`，然后执行：

```bash
python convert_data_format.py
```

如果使用当前仓库的 `data/original_data/testResult.csv`，请先修改脚本中的输入路径，或复制数据到脚本期望位置。

### 2. 训练本地 RoBERTa 分类器

```bash
python train_bert_classifier.py
```

训练完成后会生成：

```text
roberta_classifier/
├── config.json
├── model.safetensors / pytorch_model.bin
├── tokenizer.json
└── tokenizer_config.json
```

### 3. 生成中文对抗样本

先设置 DashScope API Key：

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

然后运行：

```bash
python generate_adv_sentences.py
```

如需切换生成左侧或右侧对话样本，可修改脚本中的 `SIDES` 配置。

### 4. 评估对抗攻击效果

使用 LLM 分类器：

```bash
python eval_adv_llm.py
```

使用本地 RoBERTa 分类器：

```bash
CLS_TYPE=roberta python eval_adv_llm.py
```

对话级评估示例：

```bash
python eval_adv_dialogue.py --step original --cls_type roberta
python eval_adv_dialogue.py --step adversarial --cls_type roberta
python eval_adv_dialogue.py --step calculate --attack_type asymmetric
```

## 数据说明

### 中文诈骗文本数据

`data/original_data/` 中包含中文诈骗通话数据：

- `trainResult.csv`：训练集
- `testResult.csv`：测试集
- `test_small.csv`：小规模测试样本

主要字段包括通话文本、交互策略、通话类型、诈骗标签、诈骗类型等。训练脚本主要使用：

- `specific_dialogue_content`：文本内容
- `is_fraud`：是否诈骗

### PromptAttack GLUE 数据

`data/*.json` 为 PromptAttack 生成的英文 GLUE 对抗数据，覆盖：

- `SST-2`
- `QNLI`
- `QQP`
- `RTE`
- `MNLI-m`
- `MNLI-mm`

可用于复现实验或参考原 PromptAttack 数据格式。

## 注意事项

- 当前部分脚本仍保留历史路径 `mydata/...`，运行前需要确认数据目录是否匹配。
- `requirements.txt` 较大，包含实验期间使用的完整环境，首次安装可能耗时较长。
- `generate_adv_sentences.py` 涉及外部 API 调用，建议使用环境变量管理密钥，并避免提交真实 Key。
- 训练本地 RoBERTa 分类器需要较多计算资源，推荐使用 GPU。
- 对抗样本生成会产生 API 调用成本，建议先用小样本验证流程。

## 项目来源与说明

本项目参考 ICLR 2024 论文《An LLM can Fool Itself: A Prompt-Based Adversarial Attack》的 PromptAttack 方法，并在此基础上扩展到中文诈骗文本识别场景。原始 PromptAttack 关注通用 NLP 分类任务，本仓库重点放在中文场景下的大模型生成式对抗样本、诈骗文本分类器训练以及鲁棒性评估实验。