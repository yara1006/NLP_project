import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import pandas as pd
from PromptAttack.PromptAttack import PromptAttack

attacker = PromptAttack(
    log_file="log.txt",
    API_key="sk-220f085328874e0399af946013b81ab0",
    API_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    dataset="mydata",
    # label_list=["0", "1"],
    label_list=["诈骗", "非诈骗"],
    predictor=None,
    # version="qwen-plus",
    version="qwen-max",
    # qwen3-max
    lang="zh"
)
# DEFAULT_N = 547
DEFAULT_N = 12782
def generate_for_side(side, N=DEFAULT_N):
    src_path = f'mydata/convert_Data/{side}_dialogue_sentences.csv'
    out_dir = f'mydata/new_{side}_adv'
    os.makedirs(out_dir, exist_ok=True)
    
    # Define output file path
    out_file = f'{out_dir}/{side}_dialogue_sentences_adv_sample_strategy_0.csv'

    # Read source data
    df = pd.read_csv(src_path)
    df_sample = df.head(N).copy()

    # Check for existing results to implement "breakpoint resume"
    start_index = 0
    if os.path.exists(out_file):
        try:
            df_done = pd.read_csv(out_file)
            start_index = len(df_done)
            print(f"Resuming from index {start_index}. Already processed {start_index} items.")
        except pd.errors.EmptyDataError:
            print("Output file is empty. Starting from the beginning.")
            # If file is empty, overwrite with header
            df_sample.head(0).to_csv(out_file, index=False)
    else:
        # If file does not exist, create it with header
        df_sample.head(0).to_csv(out_file, index=False)

    if start_index >= N:
        print("All items have been processed. Task is complete.")
        return

    # Process remaining data
    df_to_process = df_sample.iloc[start_index:]

    for index, row in df_to_process.iterrows():
        print(f"Processing index {index}/{N}...")
        try:
            val = str(row['fraud_label']).strip()
            y = 1 if val in ["True", "1", "true"] else 0
            
            # Only attack if the sample is labeled as fraud (y=1)
            # For non-fraud samples (y=0), we keep the original text to avoid "making a normal person look like a scammer"
            if y == 1:
                x = [["utterance_text", row['utterance_text']]]
                t_a = 0
                tau_1 = 1.0
                tau_2 = 0.40
                adv_x = attacker.attack(
                    x, y, 0, t_a, tau_1, tau_2,
                    few_shot=False, ensemble=False
                )
                adv_text = adv_x[t_a][1]
            else:
                # Non-fraud sample: keep original text
                print(f"Skipping attack for non-fraud sample at index {index}. Keeping original text.")
                adv_text = row['utterance_text']

            # Create a DataFrame for the current result and append to file
            result_df = pd.DataFrame([row.to_dict()])
            result_df['adv_text'] = adv_text
            result_df.to_csv(out_file, mode='a', header=False, index=False)
            print(f"Successfully processed and saved index {index}.")

        except Exception as e:
            print(f"An error occurred at index {index}: {e}")
            print("Stopping execution. You can restart the script to resume from this point.")
            return # Stop execution on error

# SIDES = ['left', 'right']
# SIDES = ['left', 'right']
SIDES = ['right']
for side in SIDES:
    generate_for_side(side, DEFAULT_N)
