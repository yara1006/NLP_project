

# import pandas as pd

# CONFIG = {
#     'input_file': 'mydata/testResult.csv',  # 原始完整对话数据文件
#     'output_sentences_left': 'mydata/left_dialogue_sentences.csv',  # left端输出
#     'output_sentences_right': 'mydata/right_dialogue_sentences.csv' # right端输出
# }

# def get_left_right_utterances(dialogue_text):
#     if pd.isna(dialogue_text):
#         return [], []
#     utterances = [line.strip() for line in str(dialogue_text).split('\n') if line.strip()]
#     lefts = [u.replace('left:', '').strip() for u in utterances if u.startswith('left:')]
#     rights = [u.replace('right:', '').strip() for u in utterances if u.startswith('right:')]
#     return lefts, rights

# def process_dialogue_extraction():
#     try:
#         source_data = pd.read_csv(CONFIG['input_file'])
#         print(f"数据加载成功，共{len(source_data)}条对话记录")
#     except FileNotFoundError:
#         print(f"错误：未找到输入文件 {CONFIG['input_file']}")
#         return None, None

#     extracted_left = []
#     extracted_right = []

#     for index, record in source_data.iterrows():
#         dialogue_id = f"DLG{index + 1:03d}"
#         left_utterances, right_utterances = get_left_right_utterances(record["specific_dialogue_content"])

#         for seq_num, utterance in enumerate(left_utterances, 1):
#             utterance_id = f"{dialogue_id}_L{seq_num}"
#             extracted_left.append({
#                 "utterance_id": utterance_id,
#                 "dialogue_id": dialogue_id,
#                 "speaker": "left",
#                 "utterance_text": utterance,
#                 "strategy_type": record.get("interaction_strategy", ""),
#                 "fraud_label": record.get("is_fraud", "")
#             })

#         for seq_num, utterance in enumerate(right_utterances, 1):
#             utterance_id = f"{dialogue_id}_R{seq_num}"
#             extracted_right.append({
#                 "utterance_id": utterance_id,
#                 "dialogue_id": dialogue_id,
#                 "speaker": "right",
#                 "utterance_text": utterance,
#                 "strategy_type": record.get("interaction_strategy", ""),
#                 "fraud_label": record.get("is_fraud", "")
#             })

#     left_df = pd.DataFrame(extracted_left)
#     right_df = pd.DataFrame(extracted_right)
#     left_df.to_csv(CONFIG['output_sentences_left'], index=False)
#     right_df.to_csv(CONFIG['output_sentences_right'], index=False)
#     print(f"左侧发言提取完成：共{len(left_df)}条，保存至{CONFIG['output_sentences_left']}")
#     print(f"右侧发言提取完成：共{len(right_df)}条，保存至{CONFIG['output_sentences_right']}")
#     return left_df, right_df

# if __name__ == "__main__":
#     process_dialogue_extraction()




import pandas as pd
# 读取数据
# mydata\left_dialogue_sentences_adv_sample_strategy_0.csv
# mydata\new_left_adv\left_dialogue_sentences_adv_sample_strategy_0.csv
df_left = pd.read_csv('mydata/new_left_adv/left_dialogue_sentences_adv_sample_strategy_0.csv')
# mydata\convert_Data\right_dialogue_sentences.csv
# E:\大一\科研\LLM\PromptAttack_code\mydata\new_right_adv\right_dialogue_sentences_adv_sample_strategy_0.csv
df_right = pd.read_csv('mydata/new_right_adv/right_dialogue_sentences_adv_sample_strategy_0.csv')
# mydata\new_right_adv\right_dialogue_sentences_adv_sample_strategy_0.csv

df_left['dialogue_id'] = df_left['utterance_id'].apply(lambda x: str(x).split('_')[0])
df_right['dialogue_id'] = df_right['utterance_id'].apply(lambda x: str(x).split('_')[0])

# 设置要使用的 right 数据条数
# N = 29 # 例如只使用前100条
N=518
df_right = df_right.head(N)  # 只取前N条

# 按 dialogue_id 分组，排序
dialogue_ids = sorted(set(df_left['dialogue_id']) & set(df_right['dialogue_id']))

merged_dialogues = []
for dlg_id in dialogue_ids:
    left_part = df_left[df_left['dialogue_id'] == dlg_id].sort_values('utterance_id')
    right_part = df_right[df_right['dialogue_id'] == dlg_id].sort_values('utterance_id')
    # left_texts = left_part['utterance_text'].tolist()
    left_texts = left_part['adv_text'].tolist()
    right_texts = right_part['utterance_text'].tolist()
    dialogue = []
    # 交替拼接
    for l, r in zip(left_texts, right_texts):
        dialogue.append(f"left: {l}")
        dialogue.append(f"right: {r}")
    # 补齐剩余 left
    for l in left_texts[len(right_texts):]:
        dialogue.append(f"left: {l}")
    # 补齐剩余 right
    for r in right_texts[len(left_texts):]:
        dialogue.append(f"right: {r}")
    merged_dialogues.append({
        'dialogue_id': dlg_id,
        'dialogue_text': '\n'.join(dialogue)
    })

# 保存为 CSV
merged_df = pd.DataFrame(merged_dialogues)
merged_df.to_csv('mydata/all_adv/dialogue_reconstructed_0.csv', index=False)
print(f"已完成拼接，保存至 mydata/all_adv/dialogue_reconstructed_0.csv")
