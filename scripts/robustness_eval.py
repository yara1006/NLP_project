import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import load_dataset
from src.Predict import Predict
from src.PromptAttack import PromptAttack

# It's good practice to handle system path modifications at the top
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bert_classifier_predictor import BertPredictor

def parse_args():
    """
    Parses command-line arguments in a single, consolidated function.
    """
    parser = argparse.ArgumentParser(description="Prompt-Based Adversarial Attack Evaluation")

    # --- Model and API Arguments ---
    parser.add_argument("--model", type=str, default="qwen-plus", help="The model to use for the attack (e.g., qwen-plus, gpt-3.5-turbo).")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the LLM service.")
    parser.add_argument("--api_base", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="Base URL for the LLM API.")
    parser.add_argument("--version", type=str, default=None, help="Specific version of the model for the API. If None, defaults to the value of --model.")

    # --- Dataset Arguments ---
    parser.add_argument("--dataset", type=str, default="mydata", help="要使用的数据集（例如：sst2, qnli, mydata）")
    parser.add_argument("--lang", type=str, default="zh", help="数据集的语言（'en' 或 'zh'）")
    parser.add_argument("--data_file", type=str, default=None, help="自定义数据集文件路径，用于 'mydata' 数据集")
    parser.add_argument("--num_examples", type=int, default=None, help="要运行的样本数量，默认运行所有样本")
    parser.add_argument("--batch_size", type=int, default=8, help="批量处理时的样本数量")

    # --- Attack Parameters ---
    parser.add_argument("--task_description", type=str, default=None, help="自定义任务描述，用于提示模型攻击目标")
    parser.add_argument("--tau_1", default=0.8, type=float)
    parser.add_argument("--tau_2", type=float, default=0.5, help="阈值，用于判断扰动后的文本与原始文本的语义相似度（BERTScore）")
    parser.add_argument("--pertub_type", type=str, default="word", help="扰动类型（sememe, char, word）")
    parser.add_argument("--t_a", type=int, default=0, help="索引值，指定要扰动的句子部分（0 为 left，1 为 right）")
    parser.add_argument("--few_shot", action="store_true", help="是否开启 Few-shot 学习策略")
    parser.add_argument("--ensemble", action="store_true", help="是否开启 Ensemble 策略")

    # --- Logging ---
    parser.add_argument("--attack_log_file", type=str, default="attack.db", help="攻击结果日志文件路径")
    parser.add_argument("--check_log_file", type=str, default="check.db", help="预测检查结果日志文件路径")

    return parser.parse_args()

def get_dataset(args):
    """
    根据解析的参数加载指定的数据集。
    """
    if args.dataset.lower() == "mydata":
        data_path = args.data_file or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mydata", "testResult_converted.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件未找到：{data_path}")

        df = pd.read_csv(data_path)
        df.dropna(subset=['text', 'label'], inplace=True)
        df['label'] = df['label'].astype(str)

        label_list = sorted(list(df['label'].unique()))
        label_to_int = {label: i for i, label in enumerate(label_list)}
        # Ensure the data structure matches what the attack loop expects, using integer indices for labels
        dataset = [([("sentence", row.text)], label_to_int[str(row.label)]) for row in df.itertuples()]

    else:
        dataset_name = args.dataset.lower()
        if dataset_name == "sst-2": # Alias for sst2
            dataset_name = "sst2"
            
        if "mnli" in dataset_name:
            split = "validation_matched" if dataset_name == "mnli-m" else "validation_mismatched"
            dataset_ = load_dataset("glue", "mnli", split=split)
        else:
            dataset_ = load_dataset("glue", dataset_name, split="validation")

        label_list = [str(l) for l in dataset_.features["label"]._int2str]
        # Use the integer index directly from the dataset
        dataset = [
            ([
                [key, value] for key, value in item.items() if key not in ["label", "idx"]
            ], 
            item["label"])
            for item in dataset_
        ]

    if args.num_examples is not None:
        dataset = dataset[:args.num_examples]

    # Batch the data
    loader = [dataset[i:i + args.batch_size] for i in range(0, len(dataset), args.batch_size)]
    return loader, label_list

def get_accuracy(pred, label):
    """
    计算预测准确率。
    """
    assert len(pred) == len(label)
    correct = [str(i) == str(j) for (i, j) in zip(pred, label)]
    return sum(correct) / len(label) if len(label) > 0 else 0.0

def get_ASR(pred, adv_pred, label):
    """
    计算攻击成功率（Attack Success Rate）。
    """
    assert len(pred) == len(label) and len(pred) == len(adv_pred)
    # 原始预测中正确的样本
    correct_mask = [str(pred[i]) == str(label[i]) for i in range(len(label))]
    # 对抗样本中错误的样本
    fooled_mask = [str(adv_pred[i]) != str(label[i]) for i in range(len(label))]
    
    # 攻击成功的条件：原始预测正确且对抗样本预测错误    
    successful_attack = [c and f for c, f in zip(correct_mask, fooled_mask)]
    
    num_correct_original = sum(correct_mask)
    if num_correct_original == 0:
        return 0.0
        
    return sum(successful_attack) / num_correct_original

if __name__ == "__main__":
    # 1. 解析命令行参数
    args = parse_args()

    # 2. 加载数据集并获取标签列表
    test_loader, label_list = get_dataset(args)

    # 3. 初始化预测器，现在我们有了标签列表
    version_for_api = args.version or args.model
    predictor = Predict(
        log_file=args.check_log_file,
        API_key=args.api_key,
        API_base=args.api_base,
        label_list=label_list,
        version=version_for_api,
    )

    adv_generator = PromptAttack(
        log_file=args.attack_log_file,
        API_key=args.api_key,
        API_base=args.api_base,
        dataset=args.dataset,
        label_list=label_list,
        predictor=predictor,
        version=version_for_api,
        lang=args.lang,
    )

    # 4. 初始化独立的 BERT 分类器用于评估
    bert_classifier_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bert_classifier')
    bert_predictor = BertPredictor(model_path=bert_classifier_path)

    # 5. 评估攻击效果
    # 提取原始文本和对抗文本
    label_indices = [y for batch in test_loader for _, y in batch]
    original_texts_flat = []
    adv_texts_flat = []

    # For mydata dataset, we only have one sentence part (index 0)
    # Override t_a to 0 if dataset is mydata and t_a is not explicitly set
    effective_t_a = args.t_a
    if args.dataset.lower() == "mydata" and args.t_a == 1:  # 1 is the default
        effective_t_a = 0

    # 6. 定义任务描述
    pertub_type_map = {"sememe": 0, "char": 1, "word": 3}
    perturbation_index = pertub_type_map.get(args.pertub_type, 0) # Default to 0 if not found

    # A single, clear task description for all examples
    specific_task_description = (
        f"请根据输入内容判断其标签。请从以下标签中选择一个作为答案，并且只输出标签名，不要包含其他任何解释或标点符号。"
        f"可用标签：[{', '.join(label_list)}]"
    )

    for batch in tqdm(test_loader, desc="Attacking Batches"):
        batch_x = [x for x, y in batch]
        batch_y = [y for x, y in batch]

        # Generate adversarial examples
        batch_adv_x = adv_generator.batch_attack(
            batch_x, batch_y, perturbation_index, effective_t_a, args.tau_1, args.tau_2,
            few_shot=args.few_shot, ensemble=args.ensemble, task_description=specific_task_description
        )
        # Collect original and adversarial texts for BERT evaluation
        original_texts_flat.extend([x[0][1] for x in batch_x])
        adv_texts_flat.extend([x[0] for x in batch_adv_x])
        print(f"[DEBUG] adv_texts_flat: {adv_texts_flat}")
        print(f"[DEBUG] adv_texts_flat 类型: {type(adv_texts_flat)}")

    # Get predictions from the independent BERT classifier
    bert_pred_original = bert_predictor.predict(original_texts_flat)
    bert_pred_adversarial = bert_predictor.predict(adv_texts_flat)

    # --- Debugging: Print some examples ---
    print("\n--- Sample Comparison ---")
    for i in range(min(5, len(original_texts_flat))):
        print(f"Example {i+1}:")
        print(f"  Original Text: {original_texts_flat[i]}")
        print(f"  Adversarial Text: {adv_texts_flat[i]}")
        print(f"  Ground Truth: {label_list[label_indices[i]]}")
        print(f"  Original Prediction: {label_list[bert_pred_original[i]]}")
        print(f"  Adversarial Prediction: {label_list[bert_pred_adversarial[i]]}")
    print("-------------------------\n")
    # --- End Debugging ---

    # 7. 计算并打印最终指标
    ground_truth_labels = [label_list[i] for i in label_indices]
    # 8. 转换 BERT 预测结果为标签
    bert_pred_original_labels = [label_list[pred] for pred in bert_pred_original]
    bert_pred_adversarial_labels = [label_list[pred] for pred in bert_pred_adversarial]

    # 9. 计算并打印最终指标
    natural_acc_val = get_accuracy(bert_pred_original_labels, ground_truth_labels)
    robust_acc_val = get_accuracy(bert_pred_adversarial_labels, ground_truth_labels)
    asr_val = get_ASR(bert_pred_original_labels, bert_pred_adversarial_labels, ground_truth_labels)

    print(
        f"\n[BERT EVAL] Natural Accuracy: {natural_acc_val:.4f} | "
        f"Robust Accuracy: {robust_acc_val:.4f} | "
        f"Attack Success Rate: {asr_val:.4f}"
    )


