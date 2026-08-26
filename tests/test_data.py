"""Tests for NLP_project — data conversion and evaluation utilities."""
from __future__ import annotations

import csv
import os
import tempfile

import pandas as pd
import pytest


# ── convert_data_format tests ──────────────────────────────────────────

@pytest.fixture
def sample_fraud_csv(tmp_path) -> str:
    """Create a sample fraud detection CSV file."""
    csv_file = tmp_path / "testResult.csv"
    df = pd.DataFrame({
        "specific_dialogue_content": [
            "你好，我是客服，请问有什么可以帮您",
            "恭喜您中奖了，请点击链接领取奖金",
            "您的账户存在异常，请提供银行卡号进行验证",
        ],
        "interaction_strategy": ["正常", "诈骗", "诈骗"],
        "call_type": ["咨询", "中奖", "安全"],
        "is_fraud": [False, True, True],
        "fraud_type": ["无", "中奖诈骗", "安全诈骗"],
    })
    df.to_csv(csv_file, index=False, encoding="utf-8")
    return str(csv_file)


class TestDataConversion:
    """Test data format conversion."""

    def test_convert_labels(self, sample_fraud_csv: str, tmp_path) -> None:
        """Test that labels are correctly converted from bool to int."""
        # Import the conversion function
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from scripts.convert_data_format import convert_csv

        output_file = str(tmp_path / "converted.csv")
        success = convert_csv(sample_fraud_csv, output_file)
        assert success is True

        df = pd.read_csv(output_file, encoding="utf-8")
        assert "text" in df.columns
        assert "label" in df.columns
        assert list(df["label"]) == [0, 1, 1]
        assert len(df) == 3

    def test_convert_preserves_text(self, sample_fraud_csv: str, tmp_path) -> None:
        """Test that text content is preserved during conversion."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from scripts.convert_data_format import convert_csv

        output_file = str(tmp_path / "converted.csv")
        convert_csv(sample_fraud_csv, output_file)

        df = pd.read_csv(output_file, encoding="utf-8")
        assert "你好，我是客服" in df.iloc[0]["text"]
        assert "中奖" in df.iloc[1]["text"]

    def test_convert_missing_file(self, tmp_path) -> None:
        """Test handling of missing input file."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from scripts.convert_data_format import convert_csv

        output_file = str(tmp_path / "converted.csv")
        success = convert_csv("/nonexistent/path.csv", output_file)
        assert success is False

    def test_convert_missing_columns(self, tmp_path) -> None:
        """Test handling of CSV with missing required columns."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from scripts.convert_data_format import convert_csv

        bad_csv = tmp_path / "bad.csv"
        df = pd.DataFrame({"wrong_col": ["text"]})
        df.to_csv(bad_csv, index=False)

        output_file = str(tmp_path / "converted.csv")
        success = convert_csv(str(bad_csv), output_file)
        assert success is False

    def test_create_small_test_file(self, tmp_path) -> None:
        """Test creating a small test subset."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from scripts.convert_data_format import create_small_test_file

        # Create a source file with 10 rows
        source_file = tmp_path / "source.csv"
        df = pd.DataFrame({"text": [f"sample {i}" for i in range(10)], "label": [0, 1] * 5})
        df.to_csv(source_file, index=False)

        output_file = tmp_path / "small.csv"
        success = create_small_test_file(str(source_file), str(output_file))
        assert success is True

        small_df = pd.read_csv(output_file)
        assert len(small_df) == 5


# ── LLMLogSql tests ────────────────────────────────────────────────────

# Mock dashscope before any imports from Call.py
import sys
import types
_mock_dashscope = types.ModuleType("dashscope")
_mock_dashscope.Generation = type("Generation", (), {})
sys.modules["dashscope"] = _mock_dashscope


class TestLLMLogSql:
    """Test SQLite caching layer."""

    def test_insert_and_query(self, tmp_path) -> None:
        """Test basic insert and query operations."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from src.Call import LLMLogSql

        db_file = str(tmp_path / "test_cache.db")
        log = LLMLogSql(db_file)

        log.DBInsert("test prompt", "test response")
        result = log.DBQuery("test prompt")
        assert result == "test response"

    def test_query_nonexistent(self, tmp_path) -> None:
        """Test querying a non-existent key returns None."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from src.Call import LLMLogSql

        db_file = str(tmp_path / "test_cache.db")
        log = LLMLogSql(db_file)

        result = log.DBQuery("nonexistent")
        assert result is None

    def test_upsert(self, tmp_path) -> None:
        """Test that inserting the same key updates the value."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from src.Call import LLMLogSql

        db_file = str(tmp_path / "test_cache.db")
        log = LLMLogSql(db_file)

        log.DBInsert("key", "value1")
        log.DBInsert("key", "value2")
        result = log.DBQuery("key")
        assert result == "value2"
