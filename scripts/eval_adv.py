# 评估对抗样本的分类器效果--传统分类器 BERT以及ROBERTA
import pandas as pd
from tqdm import tqdm
# from bert_classifier_predictor import BertPredictor  # 原BERT分类器相关代码
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
adv_path = 'mydata/all_adv_label/dialogue_adv_8.csv'
# model_path = 'bert_classifier'
model_path = 'roberta_classifier'
batch_size = 8
N = 100

print("开始加载原始样本数据...")
df_orig = pd.read_csv(orig_path).head(N)
orig_texts = df_orig['text'].tolist()
orig_labels = df_orig['label'].astype(str).tolist()
label_list = sorted(list(set(orig_labels)))
label_to_int = {label: i for i, label in enumerate(label_list)}
orig_int_labels = [label_to_int[label] for label in orig_labels]
print(f"原始样本数量: {len(orig_texts)}")

print("开始加载对抗样本数据...")
df_adv = pd.read_csv(adv_path).head(N)
adv_texts = df_adv['text'].tolist()
adv_labels = df_adv['label'].astype(str).tolist()
adv_int_labels = [label_to_int[label] for label in adv_labels]
print(f"对抗样本数量: {len(adv_texts)}")

print("开始加载分类器模型...")
# ======= 原BERT分类器相关代码（已注释） =======
# bert_predictor = BertPredictor(model_path=model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSequenceClassification.from_pretrained(model_path)

# ======= 新增RoBERTa分类器相关代码 =======
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def roberta_predict(texts, tokenizer, model, device='cpu', batch_size=8):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encodings = tokenizer(batch, padding=True, truncation=True, max_length=160, return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            results.extend(preds)
    return results

print("模型加载完成。")

print("正在对原始样本进行预测...")
# ======= 原BERT预测代码（已注释） =======
# orig_pred = []
# for i in tqdm(range(0, len(orig_texts), batch_size), desc="原始样本预测进度"):
#     batch = orig_texts[i:i+batch_size]
#     orig_pred.extend(bert_predictor.predict(batch))
# orig_pred_labels = [label_list[pred] for pred in orig_pred]

# ======= RoBERTa预测代码 =======
orig_pred = roberta_predict(orig_texts, tokenizer, model, device=device, batch_size=batch_size)
orig_pred_labels = [label_list[pred] for pred in orig_pred]
print("原始样本预测完成。")

print("正在对对抗样本进行预测...")
# ======= 原BERT预测代码（已注释） =======
# adv_pred = []
# for i in tqdm(range(0, len(adv_texts), batch_size), desc="对抗样本预测进度"):
#     batch = adv_texts[i:i+batch_size]
#     adv_pred.extend(bert_predictor.predict(batch))
# adv_pred_labels = [label_list[pred] for pred in adv_pred]

# ======= RoBERTa预测代码 =======
adv_pred = roberta_predict(adv_texts, tokenizer, model, device=device, batch_size=batch_size)
adv_pred_labels = [label_list[pred] for pred in adv_pred]
print("对抗样本预测完成。")

print("正在统计准确率和攻击成功率...")
orig_acc = get_accuracy(orig_pred_labels, orig_labels)
adv_acc = get_accuracy(adv_pred_labels, adv_labels)
asr = get_ASR(orig_pred_labels, adv_pred_labels, orig_labels)
print("统计完成。")

print(f"原始样本准确率: {orig_acc:.4f}")
print(f"对抗样本准确率: {adv_acc:.4f}")
print(f"攻击成功率（ASR）: {asr:.4f}")
# print("\n详细预测结果对比：")
print("\n")
# for i in range(len(orig_texts)):
#     print(f"\n样本{i+1}:")
#     print(f"原始文本: {orig_texts[i]}")
#     print(f"对抗文本: {adv_texts[i]}")
#     print(f"真实标签: {orig_labels[i]}")
#     print(f"原始预测: {orig_pred_labels[i]}")
#     print(f"对抗预测: {adv_pred_labels[i]}")
