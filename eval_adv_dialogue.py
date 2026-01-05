import os
import pandas as pd
from tqdm import tqdm
from PromptAttack.Call import LLMCall
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import argparse
import glob

# --- Configuration ---
# Input files
ORIG_LEFT_SENTENCES = 'mydata/convert_Data/left_dialogue_sentences.csv'
ORIG_RIGHT_SENTENCES = 'mydata/convert_Data/right_dialogue_sentences.csv'
# The adversarial file path will be found dynamically

# Output files
ORIG_PREDICTIONS_FILE = 'mydata/predictions/original_predictions.csv'
ADV_PREDICTIONS_FILE = 'mydata/predictions/adversarial_predictions.csv'
SYM_ADV_PREDICTIONS_FILE = 'mydata/predictions/symmetric_adversarial_predictions.csv'
FINAL_RESULTS_FILE = 'dialogue_level_evaluation_results.csv'

# --- Helper Functions (Metrics) ---
def get_accuracy(pred, label):
    """计算预测准确率"""
    assert len(pred) == len(label)
    correct = [str(i) == str(j) for (i, j) in zip(pred, label)]
    return sum(correct) / len(label) if len(label) > 0 else 0.0

def get_ASR(orig_pred, adv_pred, label):
    """计算攻击成功率 (Attack Success Rate)"""
    assert len(orig_pred) == len(label) and len(orig_pred) == len(adv_pred)
    # 找出原始预测正确的样本
    correct_mask = [str(orig_pred[i]) == str(label[i]) for i in range(len(label))]
    # 找出对抗预测错误的样本
    fooled_mask = [str(adv_pred[i]) != str(label[i]) for i in range(len(label))]
    # 攻击成功 = 原始预测正确 & 对抗预测错误
    successful_attack = [c and f for c, f in zip(correct_mask, fooled_mask)]
    num_correct_original = sum(correct_mask)
    if num_correct_original == 0:
        # 如果模型一开始就全部分类错误，那么攻击成功率为0
        return 0.0
    return sum(successful_attack) / num_correct_original

# --- New Data Assembly Function ---
def process_adv_df(df):
    """Helper to process an adversarial dataframe."""
    if 'adv_text' in df.columns:
        df.rename(columns={'adv_text': 'text'}, inplace=True)
        if 'utterance_text' in df.columns:
            df.drop(columns=['utterance_text'], inplace=True)
    elif 'utterance_text' in df.columns:
        df.rename(columns={'utterance_text': 'text'}, inplace=True)
    return df

def assemble_dialogues(mode='original'):
    """
    Assembles dialogues from sentence files.
    'original': loads original left and right sentences.
    'adversarial': loads adversarial left and original right sentences.
    'symmetric_adversarial': loads adversarial left and adversarial right sentences.
    """
    try:
        print(f"--- Assembling '{mode}' dialogues ---")
        
        # Determine and load left and right dataframes based on mode
        if mode == 'original':
            df_left = pd.read_csv(ORIG_LEFT_SENTENCES)
            if 'utterance_text' in df_left.columns:
                df_left.rename(columns={'utterance_text': 'text'}, inplace=True)
            
            df_right = pd.read_csv(ORIG_RIGHT_SENTENCES)
            if 'utterance_text' in df_right.columns:
                df_right.rename(columns={'utterance_text': 'text'}, inplace=True)

        elif mode == 'adversarial':
            adv_left_files = glob.glob('mydata/new_left_adv/*_adv_*.csv')
            if not adv_left_files:
                raise FileNotFoundError("No adversarial files found in 'mydata/new_left_adv/'.")
            print(f"Found left adversarial file: {adv_left_files[0]}")
            df_left = pd.read_csv(adv_left_files[0])
            df_left = process_adv_df(df_left)
            
            df_right = pd.read_csv(ORIG_RIGHT_SENTENCES)
            if 'utterance_text' in df_right.columns:
                df_right.rename(columns={'utterance_text': 'text'}, inplace=True)

        elif mode == 'symmetric_adversarial':
            adv_left_files = glob.glob('mydata/new_left_adv/*_adv_*.csv')
            if not adv_left_files:
                raise FileNotFoundError("No left adversarial files found in 'mydata/new_left_adv/'.")
            print(f"Found left adversarial file: {adv_left_files[0]}")
            df_left = pd.read_csv(adv_left_files[0])
            df_left = process_adv_df(df_left)

            adv_right_files = glob.glob('mydata/new_right_adv/*_adv_*.csv')
            if not adv_right_files:
                raise FileNotFoundError("No right adversarial files found in 'mydata/new_right_adv/'.")
            print(f"Found right adversarial file: {adv_right_files[0]}")
            df_right = pd.read_csv(adv_right_files[0])
            df_right = process_adv_df(df_right)

        else:
            raise ValueError("Mode must be 'original', 'adversarial', or 'symmetric_adversarial'")

        # Concatenate, sort, and group
        df_all = pd.concat([df_left, df_right], ignore_index=True)
        df_all = df_all.sort_values(by=['dialogue_id', 'utterance_id'])

        dialogues = df_all.groupby('dialogue_id')['text'].apply(lambda x: '\n'.join(x.dropna().astype(str))).reset_index()
        dialogues = dialogues.rename(columns={'text': 'dialogue_text'})

        # Merge with original data to get the fraud_label
        df_labels = pd.read_csv(ORIG_RIGHT_SENTENCES)[['dialogue_id', 'fraud_label']].drop_duplicates(subset='dialogue_id')
        df_final = pd.merge(dialogues, df_labels, on='dialogue_id')

        print(f"Successfully assembled {len(df_final)} dialogues.")
        return df_final

    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during dialogue assembly: {e}")
        raise

# --- Prediction Functions ---
def build_cls_prompt(text, label_list):
    """构建用于分类的Prompt"""
    return f"请判断下面这段通话记录属于'{label_list[0]}'或'{label_list[1]}'中的哪一类。请只输出这两个类别名称中的一个，不要包含任何解释或多余的文字。\n\n通话记录：\n{text}"

def qwen_predict(texts, label_list, llm, batch_size=8):
    """使用Qwen LLM API进行批量预测"""
    results = []
    sorted_labels_with_idx = sorted(enumerate(label_list), key=lambda x: len(x[1]), reverse=True)
    
    for i in tqdm(range(0, len(texts), batch_size), desc="LLM API Predicting"):
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
                results.append(0) # Default to non-fraud
    return results

def roberta_predict(texts, tokenizer, model, device, batch_size=8):
    """使用本地RoBERTa模型进行批量预测"""
    preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="RoBERTa Predicting"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            preds.extend(batch_preds)
    return preds

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Run dialogue-level evaluation in steps.")
    parser.add_argument('--step', required=True, choices=['original', 'adversarial', 'symmetric_adversarial', 'calculate'], help="The step to execute.")
    parser.add_argument('--attack_type', default='asymmetric', choices=['asymmetric', 'symmetric'], help="For 'calculate' step, specifies which attack to evaluate.")
    parser.add_argument('--cls_type', default=os.environ.get('CLS_TYPE', 'llm'), choices=['llm', 'roberta'], help="Classifier type.")
    parser.add_argument('--model_path', default='roberta_classifier', help="Path to local RoBERTa model.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for prediction.")
    parser.add_argument('--limit', type=int, default=None, help="Limit the number of samples to process.")
    args = parser.parse_args()

    # --- Prediction Steps ---
    if args.step in ['original', 'adversarial', 'symmetric_adversarial']:
        # 1. Assemble Data
        df = assemble_dialogues(mode=args.step)
        if df is None:
            return
        if args.limit:
            df = df.head(args.limit)
            print(f"Processing a limit of {len(df)} samples.")

        texts = df['dialogue_text'].tolist()
        
        # 2. Load Classifier
        print(f"--- Loading classifier: {args.cls_type.upper()} ---")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        label_list = ['非诈骗', '诈骗']
        predictions = []

        if args.cls_type == 'roberta':
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
                model.to(device)
                model.eval()
                print("Local RoBERTa model loaded.")
                predictions = roberta_predict(texts, tokenizer, model, device, batch_size=args.batch_size)
            except Exception as e:
                print(f"Error loading RoBERTa model: {e}")
                return
        else: # LLM
            try:
                api_key = os.environ.get("DASHSCOPE_API_KEY", "11")
                if not api_key:
                    raise ValueError("DASHSCOPE_API_KEY environment variable not set.")
                version = "qwen-max"
                llm = LLMCall(log_file=f"eval_{args.step}_log.sqlite", API_key=api_key, version=version)
                print("LLM API initialized.")
                predictions = qwen_predict(texts, label_list, llm, batch_size=args.batch_size)
            except Exception as e:
                print(f"Error initializing LLM API: {e}")
                return
        
        # 3. Save Results
        df['prediction_index'] = predictions
        df['prediction_label'] = [label_list[p] for p in predictions]
        
        if args.step == 'original':
            output_file = ORIG_PREDICTIONS_FILE
        elif args.step == 'adversarial':
            output_file = ADV_PREDICTIONS_FILE
        else: # symmetric_adversarial
            output_file = SYM_ADV_PREDICTIONS_FILE

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Predictions for '{args.step}' step saved to: {output_file}")

    # --- Calculation Step ---
    elif args.step == 'calculate':
        if args.attack_type == 'asymmetric':
            adv_file_path = ADV_PREDICTIONS_FILE
            attack_name = "Asymmetric Attack"
        else: # symmetric
            adv_file_path = SYM_ADV_PREDICTIONS_FILE
            attack_name = "Symmetric Attack"

        print(f"--- Calculating Final ASR for {attack_name} ---")
        try:
            df_orig = pd.read_csv(ORIG_PREDICTIONS_FILE)
            df_adv = pd.read_csv(adv_file_path)
        except FileNotFoundError as e:
            print(f"Error: Prediction files not found. Please run 'original' and '{args.attack_type}_adversarial' steps first. {e}")
            return
            
        # Merge results
        df_merged = pd.merge(
            df_orig[['dialogue_id', 'dialogue_text', 'fraud_label', 'prediction_label']],
            df_adv[['dialogue_id', 'dialogue_text', 'prediction_label']],
            on='dialogue_id',
            suffixes=('_orig', '_adv')
        )
        
        # Normalize labels
        def normalize_labels(labels):
            normalized = []
            for l in labels:
                if str(l).lower() in ['1', 'true', '诈骗']:
                    normalized.append('诈骗')
                else:
                    normalized.append('非诈骗')
            return normalized

        true_labels = normalize_labels(df_merged['fraud_label'].tolist())
        orig_pred_labels = df_merged['prediction_label_orig'].tolist()
        adv_pred_labels = df_merged['prediction_label_adv'].tolist()

        # Calculate metrics
        orig_acc = get_accuracy(orig_pred_labels, true_labels)
        adv_acc = get_accuracy(adv_pred_labels, true_labels)
        asr = get_ASR(orig_pred_labels, adv_pred_labels, true_labels)

        print("\n" + "="*20 + f" {attack_name} Results " + "="*20)
        print(f"Original Dialogue Accuracy: {orig_acc:.4f}")
        print(f"Adversarial Dialogue Accuracy ({args.attack_type}): {adv_acc:.4f}")
        print(f"Attack Success Rate (ASR) ({args.attack_type}): {asr:.4f}")
        print("="* (42 + len(attack_name)) + "\n")

        # Save final detailed results
        final_results_df = pd.DataFrame({
            'dialogue_id': df_merged['dialogue_id'],
            'original_text': df_merged['dialogue_text_orig'],
            'adversarial_text': df_merged['dialogue_text_adv'],
            'true_label': true_labels,
            'original_prediction': orig_pred_labels,
            'adversarial_prediction': adv_pred_labels
        })
        
        results_filename = f"dialogue_level_evaluation_results_{args.attack_type}.csv"
        final_results_df.to_csv(results_filename, index=False, encoding='utf-8-sig')
        print(f"Detailed final results for {attack_name} saved to: {results_filename}")

if __name__ == "__main__":
    main()
