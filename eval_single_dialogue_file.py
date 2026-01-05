
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import sys
from PromptAttack.Call import LLMCall

# --- 配置 ---
CONFIG = {
    "data_file": "mydata/all_adv_label/dialogue_adv_0.csv",
    "n_samples": None,  # 设置为None以处理整个文件，或设置为一个整数以处理部分样本
    "llm_config": {
        "provider": "dashscope",
        "model_name": "qwen-max",
        "api_key": os.environ.get("DASHSCOPE_API_KEY")
    },
    "label_list": ['非诈骗', '诈骗'],
    "batch_size": 8,
}

def build_cls_prompt(text, label_list):
    """构建用于分类的prompt"""
    return f"请判断下面这段通话记录属于'{label_list[0]}'或'{label_list[1]}'中的哪一类。请只输出这两个类别名称中的一个，不要包含任何解释或多余的文字。\n\n通话记录：\n{text}"

def qwen_predict(llm, texts, label_list, batch_size):
    """使用Qwen模型进行批量预测"""
    results = []
    # 将标签按长度降序排序，优先匹配更长的标签（例如“非诈骗”优先于“诈骗”）
    sorted_labels_with_idx = sorted(enumerate(label_list), key=lambda x: len(x[1]), reverse=True)
    
    for i in tqdm(range(0, len(texts), batch_size), desc="LLM API Predicting"):
        batch_texts = texts[i:i+batch_size]
        prompts = [build_cls_prompt(t, label_list) for t in batch_texts]
        
        # 这个示例假设llm.query可以处理批量请求或我们在此循环
        # 在这个包装器中，我们一次只发送一个请求
        for prompt in prompts:
            try:
                response = llm.query(prompt).strip()
                # 清理和规范化API返回的文本
                clean_response = response.replace(' ', '').replace('\n', '').replace('\r', '').replace('。', '').replace('，', '').replace('"', '').replace("'", '').strip()
                
                matched = False
                for idx, label in sorted_labels_with_idx:
                    if clean_response == label:
                        results.append(idx)
                        matched = True
                        break
                
                if not matched:
                    # 如果没有精确匹配，可以设置一个默认值，例如最常见的类别 '非诈骗' (0)
                    # print(f"Warning: LLM response '{response}' did not exactly match any label. Defaulting to '{label_list[0]}'.")
                    results.append(0)
            except Exception as e:
                print(f"Error during API call: {e}")
                results.append(0) # or handle error appropriately
                
    return results

def main():
    """主函数"""
    # seed_everything(42) # 暂时移除

    # --- 1. 初始化LLM ---
    print("--- 初始化LLM分类器 ---")
    # 修正LLMCall的初始化方式
    llm = LLMCall(
        log_file="eval_single_file_log.sqlite", 
        API_key=CONFIG["llm_config"]["api_key"],
        version=CONFIG["llm_config"]["model_name"]
    )
    
    # --- 2. 加载数据 ---
    data_file = CONFIG["data_file"]
    print(f"--- 开始加载评估数据: {data_file} ---")
    try:
        df = pd.read_csv(data_file)
        if CONFIG["n_samples"] is not None:
            df = df.head(CONFIG["n_samples"])
        print(f"成功加载 {len(df)} 条数据。")
    except FileNotFoundError:
        print(f"错误：评估数据文件 '{data_file}' 未找到！")
        return

    # --- 3. 准备标签 ---
    # 假设 'label' 列已经是 0 和 1
    if 'label' not in df.columns or 'text' not in df.columns:
        print(f"错误：文件 {data_file} 必须包含 'text' 和 'label' 列。")
        return
        
    true_labels = df['label'].tolist()
    texts = df['text'].tolist()

    # --- 4. 进行预测 ---
    print("--- 开始使用LLM进行预测 ---")
    predicted_labels = qwen_predict(llm, texts, CONFIG["label_list"], CONFIG["batch_size"])

    # --- 5. 计算和打印结果 ---
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    print("\n--- 评估结果 ---")
    print(f"处理文件: {data_file}")
    print(f"样本总数: {len(texts)}")
    print(f"模型分类准确率 (Accuracy): {accuracy:.4f}")
    print("--- 评估完成 ---")

    # --- 6. (可选) 保存预测结果 ---
    df['predicted_label_id'] = predicted_labels
    df['predicted_label_name'] = [CONFIG['label_list'][i] for i in predicted_labels]
    output_filename = 'mydata/predictions/dialogue_adv_0_predictions.csv'
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"详细预测结果已保存至: {output_filename}")


if __name__ == "__main__":
    main()
