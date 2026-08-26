#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import csv

def convert_csv(input_file: str, output_file: str) -> bool:
    """Convert fraud detection CSV from original format to text/label format.

    Args:
        input_file: Path to input CSV with specific_dialogue_content and is_fraud columns.
        output_file: Path to output CSV with text and label columns.

    Returns:
        True if conversion succeeded, False otherwise.
    """
    if not os.path.exists(input_file):
        print(f"找不到输入文件: {input_file}")
        return False

    try:
        df = pd.read_csv(input_file, encoding="utf-8")
        print(f"成功读取原始文件，共 {len(df)} 条数据")

        required_columns = ["specific_dialogue_content", "is_fraud"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"缺少必要的列: {missing_columns}")
            return False

        converted_df = pd.DataFrame({
            "text": df["specific_dialogue_content"],
            "label": df["is_fraud"].fillna(False).astype(int),
        })

        converted_df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"成功转换数据格式，保存到: {output_file}")
        return True

    except Exception as e:
        print(f"转换过程中出错: {e}")
        return False


def convert_mydata_format():
    """Convert mydata testResult.csv to standard format (backward compatible)."""
    input_file = os.path.join("mydata", "testResult.csv")
    output_file = os.path.join("mydata", "testResult_converted.csv")
    return convert_csv(input_file, output_file)

def create_small_test_file(input_file: str, output_file: str, n: int = 5) -> bool:
    """Create a small test subset from a converted CSV file.

    Args:
        input_file: Path to the full converted CSV.
        output_file: Path for the small test CSV.
        n: Number of rows to include (default 5).

    Returns:
        True if successful, False otherwise.
    """
    if not os.path.exists(input_file):
        print(f"找不到转换后的文件: {input_file}")
        return False

    try:
        df = pd.read_csv(input_file, encoding="utf-8")
        small_df = df.head(n)
        small_df.to_csv(output_file, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
        print(f"成功创建小规模测试文件: {output_file} ({len(small_df)} 条数据)")
        return True
    except Exception as e:
        print(f"创建测试文件出错: {e}")
        return False

if __name__ == "__main__":
    print("开始转换数据格式...")

    if convert_mydata_format():
        print("\n创建小规模测试文件...")
        if create_small_test_file(
            os.path.join("mydata", "testResult_converted.csv"),
            os.path.join("mydata", "test_small.csv"),
        ):
            print("\n数据格式转换完成！")
        else:
            print("创建测试文件失败")
    else:
        print("数据格式转换失败")