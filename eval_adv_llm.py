# 评估对抗样本的分类器效果--传统分类器 BERT以及ROBERTA
# 自攻击，自身生成对抗样本，攻击自身
import os
import pandas as pd
from tqdm import tqdm
from PromptAttack.Call import LLMCall
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def get_accuracy(pred, label):
    assert len(pred) == len(label)
    correct = [str(i) == str(j) for (i, j) in zip(pred, label)]
    return sum(correct) / len(label) if len(label) > 0 else 0.0

def get_ASR(orig_pred, adv_pred, label):
    assert len(orig_pred) == len(label) and len(orig_pred) == len(adv_pred)
    correct_mask = [str(orig_pred[i]) == str(label[i]) for i in range(len(label))]
    fooled_mask = [str(adv_pred[i]) != str(label[i]) for i in range(len(label))]
    successful_attack = [c and f for c, f in zip(correct_mask, fooled_mask)]
    num_correct_original = sum(correct_mask)
    if num_correct_original == 0:
        return 0.0
    return sum(successful_attack) / num_correct_original

# 路径设置
orig_path = 'mydata/convert_Data/testResult_converted.csv'
# adv_path = 'mydata/all_adv_label/dialogue_adv_0.csv'
# adv_path='mydata/all_adv_label/dialogue_adv_1.csv'
adv_path = 'mydata/new_left_adv/left_dialogue_sentences_adv_sample_strategy_0.csv'
model_path = 'roberta_classifier'
batch_size = 8
N = 547
CLS_TYPE = os.environ.get('CLS_TYPE', 'llm')

print("开始加载原始样本数据...")
df_orig = pd.read_csv(orig_path).head(N)
orig_texts = df_orig['text'].tolist()
orig_labels = df_orig['label'].astype(str).str.strip().tolist()

# 统一标签语义为中文：非诈骗/诈骗
label_list = ['非诈骗', '诈骗']
label_to_int = {label: i for i, label in enumerate(label_list)}

# 将原始标签转换为中文语义
if set(orig_labels) <= {'0', '1'}:
    orig_labels = ['非诈骗' if l == '0' else '诈骗' for l in orig_labels]
elif set(orig_labels) <= {'False', 'True'}:
    orig_labels = ['非诈骗' if l == 'False' else '诈骗' for l in orig_labels]
elif set(orig_labels) <= {'非诈骗', '诈骗'}:
    pass
else:
    orig_labels = ['非诈骗' if l.lower() in ['0', 'false', 'non-fraud', '非诈骗'] else '诈骗' for l in orig_labels]

orig_int_labels = [label_to_int[label] for label in orig_labels]
print(f"原始样本数量: {len(orig_texts)}")

print("开始加载对抗样本数据...")
df_adv = pd.read_csv(adv_path).head(N)
print(f"对抗样本列名: {df_adv.columns.tolist()}")

# 自动适配列名
text_col = 'adv_text' if 'adv_text' in df_adv.columns else 'text'
label_col = 'fraud_label' if 'fraud_label' in df_adv.columns else 'label'

adv_texts = df_adv[text_col].astype(str).tolist()
adv_labels = df_adv[label_col].astype(str).str.strip().tolist()

# 将对抗标签转换为中文语义
if set(adv_labels) <= {'0', '1'}:
    adv_labels = ['非诈骗' if l == '0' else '诈骗' for l in adv_labels]
elif set(adv_labels) <= {'False', 'True'}:
    adv_labels = ['非诈骗' if l == 'False' else '诈骗' for l in adv_labels]
elif set(adv_labels) <= {'非诈骗', '诈骗'}:
    pass
else:
    adv_labels = ['非诈骗' if l.lower() in ['0', 'false', 'non-fraud', '非诈骗'] else '诈骗' for l in adv_labels]

adv_int_labels = [label_to_int[label] for label in adv_labels]
print(f"对抗样本数量: {len(adv_texts)}")

if 'utterance_id' in df_adv.columns and 'adv_text' in df_adv.columns:
    print("使用逐句配对评估模式")
    df_adv_full = pd.read_csv(adv_path)
    df_adv_full = df_adv_full.rename(columns={'fraud_label': 'fraud_label_adv'})
    # 根据对抗文件的端侧选择左/右原始逐句文件
    orig_side_path = 'mydata/convert_Data/left_dialogue_sentences.csv'
    try:
        first_sid = df_adv_full['speaker'].iloc[0] if 'speaker' in df_adv_full.columns else None
        has_right_id = df_adv_full['utterance_id'].astype(str).str.contains('_R').any()
        if first_sid == 'right' or has_right_id:
            orig_side_path = 'mydata/convert_Data/right_dialogue_sentences.csv'
    except Exception:
        pass
    df_orig_ut = pd.read_csv(orig_side_path)
    df_orig_ut = df_orig_ut.rename(columns={'fraud_label': 'fraud_label_orig'})
    df_merge = pd.merge(
        df_adv_full[['utterance_id', 'adv_text', 'fraud_label_adv']],
        df_orig_ut[['utterance_id', 'utterance_text', 'fraud_label_orig']],
        on='utterance_id', how='inner'
    )
    df_merge = df_merge.head(N)
    orig_texts = df_merge['utterance_text'].astype(str).tolist()
    adv_texts = df_merge['adv_text'].astype(str).tolist()
    orig_labels = df_merge['fraud_label_orig'].astype(str).str.strip().tolist()
    adv_labels = df_merge['fraud_label_adv'].astype(str).str.strip().tolist()
    if set(orig_labels) <= {'0', '1'}:
        orig_labels = ['非诈骗' if l == '0' else '诈骗' for l in orig_labels]
    elif set(orig_labels) <= {'False', 'True'}:
        orig_labels = ['非诈骗' if l == 'False' else '诈骗' for l in orig_labels]
    elif set(orig_labels) <= {'非诈骗', '诈骗'}:
        pass
    else:
        orig_labels = ['非诈骗' if l.lower() in ['0', 'false', 'non-fraud', '非诈骗'] else '诈骗' for l in orig_labels]
    if set(adv_labels) <= {'0', '1'}:
        adv_labels = ['非诈骗' if l == '0' else '诈骗' for l in adv_labels]
    elif set(adv_labels) <= {'False', 'True'}:
        adv_labels = ['非诈骗' if l == 'False' else '诈骗' for l in adv_labels]
    elif set(adv_labels) <= {'非诈骗', '诈骗'}:
        pass
    else:
        adv_labels = ['非诈骗' if l.lower() in ['0', 'false', 'non-fraud', '非诈骗'] else '诈骗' for l in adv_labels]
    orig_int_labels = [label_to_int[label] for label in orig_labels]
    adv_int_labels = [label_to_int[label] for label in adv_labels]
    same_ratio = sum([o == a for o, a in zip(orig_texts, adv_texts)]) / len(orig_texts) if len(orig_texts) > 0 else 0.0
    print(f"对抗样本与原文完全相同比例: {same_ratio:.4f}")

print("开始加载分类器模型...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if CLS_TYPE == 'roberta':
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
else:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "11")
    api_base = os.environ.get("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    version = "qwen3-max"
    llm = LLMCall(log_file="eval_log.sqlite", API_key=api_key, API_base=api_base, version=version)

def build_cls_prompt(text, label_list):
    return f"请判断下面文本属于'{label_list[0]}'或'{label_list[1]}'中的哪一类。只输出这两个词之一，不要解释。\n文本：{text}"

def qwen_predict(texts, label_list, batch_size=8):
    results = []
    sorted_labels_with_idx = sorted(enumerate(label_list), key=lambda x: len(x[1]), reverse=True)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        prompts = [build_cls_prompt(t, label_list) for t in batch]
        for p in prompts:
            r = llm.query(p).strip()
            rc = r.replace(' ', '').replace('\n', '').replace('\r', '').replace('。', '').replace('，', '').replace('"', '').replace("'", '').strip()
            matched = False
            for idx, label in sorted_labels_with_idx:
                if rc == label:
                    results.append(idx)
                    matched = True
                    break
            if not matched:
                results.append(0)
    return results

def roberta_predict(texts, tokenizer, model, device, batch_size=8):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=160, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            preds.extend(batch_preds)
    return preds

print("正在对原始样本进行预测...")
if CLS_TYPE == 'roberta':
    orig_pred = roberta_predict(orig_texts, tokenizer, model, device=device, batch_size=batch_size)
else:
    orig_pred = qwen_predict(orig_texts, label_list, batch_size=batch_size)
orig_pred_labels = [label_list[p] for p in orig_pred]
print("原始样本预测完成。")

print("正在对对抗样本进行预测...")
if CLS_TYPE == 'roberta':
    adv_pred = roberta_predict(adv_texts, tokenizer, model, device=device, batch_size=batch_size)
else:
    adv_pred = qwen_predict(adv_texts, label_list, batch_size=batch_size)
adv_pred_labels = [label_list[p] for p in adv_pred]
print("对抗样本预测完成。")

print("正在统计准确率和攻击成功率...")
orig_acc = get_accuracy(orig_pred_labels, orig_labels)
adv_acc = get_accuracy(adv_pred_labels, adv_labels)
asr = get_ASR(orig_pred_labels, adv_pred_labels, orig_labels)
print("统计完成。")
print("\n")
for i in range(len(orig_texts)):
    print(f"\n样本{i+1}:")
    print(f"原始文本: {orig_texts[i]}")
    print(f"对抗文本: {adv_texts[i]}")
    print(f"真实标签: {orig_labels[i]}")
    print(f"原始预测: {orig_pred_labels[i]}")
    print(f"对抗预测: {adv_pred_labels[i]}")



print(f"原始样本准确率: {orig_acc:.4f}")
print(f"对抗样本准确率: {adv_acc:.4f}")
print(f"攻击成功率（ASR）: {asr:.4f}")
print("\n详细预测结果对比：")

