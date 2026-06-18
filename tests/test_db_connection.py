"""数据库连接上传服务测试。"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pandas as pd
import pytest

from services.db_connection_service import load_from_sql


def test_sqlite_connection():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        from sqlalchemy import create_engine

        engine = create_engine(f"sqlite:///{db_path}")
        df.to_sql("test_table", engine, index=False)
        engine.dispose()

        result = load_from_sql("sqlite", {"file_path": db_path}, "SELECT * FROM test_table")
        assert len(result) == 3
        assert list(result.columns) == ["a", "b"]
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_unsupported_connection_type():
    with pytest.raises(ValueError, match="不支持的数据库类型"):
        load_from_sql("oracle", {}, "SELECT 1")
