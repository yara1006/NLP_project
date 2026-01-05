#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import csv

def convert_mydata_format():
    """
    将原始数据格式转换为代码期望的格式
    原始格式: specific_dialogue_content, interaction_strategy, call_type, is_fraud, fraud_type
    目标格式: text, label
    """
    
    # 输入文件路径
    input_file = os.path.join("mydata", "testResult.csv")
    output_file = os.path.join("mydata", "testResult_converted.csv")
    
    if not os.path.exists(input_file):
        print(f"❌ 找不到输入文件: {input_file}")
        return False
    
    try:
        # 读取原始CSV文件
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"✅ 成功读取原始文件，共 {len(df)} 条数据")
        
        # 检查必要的列是否存在
        required_columns = ['specific_dialogue_content', 'is_fraud']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ 缺少必要的列: {missing_columns}")
            return False
        
        # 转换数据格式
        # 使用 specific_dialogue_content 作为 text
        # 使用 is_fraud 作为 label (True -> 1, False -> 0)
        # 处理 NaN 值：将空值填充为 False (0)
        converted_df = pd.DataFrame({
            'text': df['specific_dialogue_content'],
            'label': df['is_fraud'].fillna(False).astype(int)  # 将布尔值转换为整数 (True=1, False=0)
        })
        
        # 保存转换后的文件
        converted_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ 成功转换数据格式，保存到: {output_file}")
        print(f"   样本数据预览:")
        for i in range(min(3, len(converted_df))):
            print(f"   {i+1}. 标签: {converted_df.iloc[i]['label']}, 文本长度: {len(converted_df.iloc[i]['text'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 转换过程中出错: {e}")
        return False

def create_small_test_file():
    """创建小规模测试文件"""
    input_file = os.path.join("mydata", "testResult_converted.csv")
    output_file = os.path.join("mydata", "test_small.csv")
    
    if not os.path.exists(input_file):
        print(f"❌ 找不到转换后的文件: {input_file}")
        return False
    
    try:
        # 读取转换后的文件
        df = pd.read_csv(input_file, encoding='utf-8')
        
        # 取前5行数据（不包含表头）
        small_df = df.head(5)
        small_df.to_csv(output_file, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
        
        print(f"✅ 成功创建小规模测试文件: {output_file}")
        print(f"   包含 {len(small_df)} 条数据")
        
        return True
        
    except Exception as e:
        print(f"❌ 创建测试文件出错: {e}")
        return False

if __name__ == "__main__":
    print("🔄 开始转换数据格式...")
    
    if convert_mydata_format():
        print("\n🔄 创建小规模测试文件...")
        if create_small_test_file():
            print("\n🎉 数据格式转换完成！")
            print("   现在您可以运行小规模测试了:")
            print("   python run_qwen.py")
            print("   然后选择选项 2")
        else:
            print("❌ 创建测试文件失败")
    else:
        print("❌ 数据格式转换失败")