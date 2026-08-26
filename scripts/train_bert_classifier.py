
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import numpy as np

# 1. 定义数据集类
class ScamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2. 训练和评估函数
def train_epoch(model, data_loader, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

# 3. 主程序
if __name__ == '__main__':
    # 参数设置
    # PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
    PRE_TRAINED_MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'
    MAX_LEN = 160
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    # MODEL_SAVE_PATH = './bert_classifier'
    MODEL_SAVE_PATH = './roberta_classifier'

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    df_train = pd.read_csv('./mydata/original_data/trainResult.csv')
    # mydata\original_data\trainResult.csv
    df_test = pd.read_csv('./mydata/original_data/testResult.csv')

    # 数据清洗：删除标签或文本为空的行，并转换标签类型
    df_train.dropna(subset=['specific_dialogue_content', 'is_fraud'], inplace=True)
    df_test.dropna(subset=['specific_dialogue_content', 'is_fraud'], inplace=True)
    df_train['is_fraud'] = df_train['is_fraud'].astype(int)
    df_test['is_fraud'] = df_test['is_fraud'].astype(int)


    # 初始化Tokenizer
    # tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    # 创建DataLoader
    train_dataset = ScamDataset(
        texts=df_train.specific_dialogue_content.to_numpy(),
        labels=df_train.is_fraud.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = ScamDataset(
        texts=df_test.specific_dialogue_content.to_numpy(),
        labels=df_test.is_fraud.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # # 加载模型
    # model = BertForSequenceClassification.from_pretrained(
    #     PRE_TRAINED_MODEL_NAME,
    #     num_labels=2 # 诈骗 vs 非诈骗
    # )
    # 加载RoBERTa模型
    model = AutoModelForSequenceClassification.from_pretrained(
        PRE_TRAINED_MODEL_NAME,
        num_labels=2 # 诈骗 vs 非诈骗
    )
    model = model.to(device)

    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            optimizer,
            device,
            len(df_train)
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            test_data_loader,
            device,
            len(df_test)
        )
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()

    # 保存模型
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")





