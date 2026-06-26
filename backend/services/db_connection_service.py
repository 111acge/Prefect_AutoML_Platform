# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""数据库连接上传服务。

支持 MySQL、PostgreSQL、ClickHouse、SQLite 通过 SQL 查询导入数据集。
"""

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from i18n import _


def load_from_sql(
    connection_type: str,
    connection_params: Dict[str, Any],
    query: str,
) -> pd.DataFrame:
    """从 SQL 数据源加载 DataFrame。

    Args:
        connection_type: mysql / postgresql / clickhouse / sqlite
        connection_params: host / port / database / username / password / file_path
        query: SQL 查询语句

    Returns:
        查询结果 DataFrame
    """
    connection_type = connection_type.lower()

    if connection_type == "sqlite":
        return _load_sqlite(connection_params, query)
    if connection_type == "mysql":
        return _load_with_sqlalchemy(connection_params, query, "mysql+pymysql")
    if connection_type == "postgresql":
        return _load_with_sqlalchemy(connection_params, query, "postgresql+psycopg2")
    if connection_type == "clickhouse":
        return _load_clickhouse(connection_params, query)

    raise ValueError(_("db.unsupported_type", type=connection_type))


def _load_sqlite(connection_params: Dict[str, Any], query: str) -> pd.DataFrame:
    file_path = connection_params.get("file_path")
    if not file_path:
        raise ValueError(_("db.sqlite_requires_path"))
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{file_path}")
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df
    finally:
        engine.dispose()


def _load_with_sqlalchemy(
    connection_params: Dict[str, Any],
    query: str,
    dialect: str,
) -> pd.DataFrame:
    try:
        from sqlalchemy import create_engine
    except ImportError as e:
        raise RuntimeError(_("db.sqlalchemy_not_installed")) from e

    host = connection_params.get("host", "localhost")
    port = connection_params.get("port")
    database = connection_params.get("database", "")
    username = connection_params.get("username", "")
    password = connection_params.get("password", "")

    if dialect == "mysql+pymysql":
        try:
            import pymysql  # noqa: F401
        except ImportError as e:
            raise RuntimeError(_("db.mysql_driver_missing")) from e
    elif dialect == "postgresql+psycopg2":
        try:
            import psycopg2  # noqa: F401
        except ImportError as e:
            raise RuntimeError(_("db.postgres_driver_missing")) from e

    url = f"{dialect}://{username}:{password}@{host}:{port}/{database}"
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df
    finally:
        engine.dispose()


def _load_clickhouse(connection_params: Dict[str, Any], query: str) -> pd.DataFrame:
    try:
        import clickhouse_connect
    except ImportError as e:
        raise RuntimeError(_("db.clickhouse_driver_missing")) from e

    host = connection_params.get("host", "localhost")
    port = connection_params.get("port", 8123)
    database = connection_params.get("database", "default")
    username = connection_params.get("username", "default")
    password = connection_params.get("password", "")

    client = clickhouse_connect.get_client(
        host=host,
        port=int(port),
        database=database,
        username=username,
        password=password,
    )
    return client.query_df(query)


def build_connection_display_name(connection_type: str, connection_params: Dict[str, Any]) -> str:
    """生成数据集显示名称。"""
    if connection_type.lower() == "sqlite":
        file_path = Path(connection_params.get("file_path", "sqlite.db"))
        return f"sqlite:{file_path.name}"
    host = connection_params.get("host", "localhost")
    database = connection_params.get("database", "")
    return f"{connection_type}:{host}/{database}"
