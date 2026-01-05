# 生成统一格式的对抗样本数据集
import pandas as pd

# 路径设置
orig_path = 'mydata/convert_Data/testResult_converted.csv'
adv_path = 'mydata/all_adv/dialogue_reconstructed_1.csv'
adv_out_path = 'mydata/all_adv_label/dialogue_adv_1.csv'
# adv_path = 'mydata/all_adv/dialogue_reconstructed_2.csv'
# adv_out_path = 'mydata/all_adv_label/dialogue_adv_2.csv'
# adv_path = 'mydata/all_adv/dialogue_reconstructed_1.csv'
# adv_out_path = 'mydata/all_adv_label/dialogue_adv_1.csv'
# adv_path = 'mydata/all_adv/dialogue_reconstructed_3.csv'
# adv_out_path = 'mydata/all_adv_label/dialogue_adv_3.csv'
# adv_path = 'mydata/all_adv/dialogue_reconstructed_4.csv'
# adv_out_path = 'mydata/all_adv_label/dialogue_adv_4.csv'
# adv_path = 'mydata/all_adv/dialogue_reconstructed_5.csv'
# adv_out_path = 'mydata/all_adv_label/dialogue_adv_5.csv'

# adv_path = 'mydata/all_adv/dialogue_reconstructed_6.csv'
# adv_out_path = 'mydata/all_adv_label/dialogue_adv_6.csv'

# adv_path = 'mydata/all_adv/dialogue_reconstructed_7.csv'
# adv_out_path = 'mydata/all_adv_label/dialogue_adv_7.csv'
# adv_path = 'mydata/all_adv/dialogue_reconstructed_8.csv'
# adv_out_path = 'mydata/all_adv_label/dialogue_adv_8.csv'
# 1. 读取原始样本
df_orig = pd.read_csv(orig_path)  # 包含 text,label
# 给原始样本加 dialogue_id（假设顺序一一对应）
df_orig['dialogue_id'] = ['DLG{:03d}'.format(i+1) for i in range(len(df_orig))]

# 2. 读取对抗样本
df_adv = pd.read_csv(adv_path)  # 包含 dialogue_id, dialogue_text

# 3. 合并标签到对抗样本
df_adv = pd.merge(df_adv, df_orig[['dialogue_id', 'label']], on='dialogue_id', how='left')

# 4. 改名为 text,label 格式，保存新文件
df_adv.rename(columns={'dialogue_text': 'text'}, inplace=True)
df_adv[['text', 'label']].to_csv(adv_out_path, index=False)

print(f'已生成带标签的对抗样本文件: {adv_out_path}')