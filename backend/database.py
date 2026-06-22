"""数据库连接与会话管理。"""

import asyncio
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import inspect
from sqlalchemy.dialects import sqlite
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from config import settings

# 创建异步引擎
engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
    poolclass=NullPool,
)

# 异步会话工厂
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# ORM 基类
Base = declarative_base()


def _sqlite_url_to_path(url: str) -> Path:
    """把 sqlalchemy aiosqlite URL 转换为本地文件路径。"""
    # 例如 sqlite+aiosqlite:///F:/KIMIWorkSpace/data/db.sqlite
    prefix = "sqlite+aiosqlite:///"
    if url.startswith(prefix):
        return Path(url[len(prefix):]).resolve()
    # 兼容 sqlite:/// 或相对路径
    parsed = urlparse(url.replace("+aiosqlite", ""))
    return Path(parsed.path).resolve()


def _migrate_sqlite(db_url: str) -> None:
    """对 SQLite 执行轻量级自动迁移：只添加 ORM 中已有但表里缺失的列。

    生产环境建议使用 Alembic；这里仅用于开发/本地环境，避免改模型后手动删库。
    """
    path = _sqlite_url_to_path(db_url)
    if not path.exists():
        return

    conn = sqlite3.connect(str(path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cur.fetchall()}

        for table_name, table in Base.metadata.tables.items():
            if table_name not in existing_tables:
                continue
            cur.execute(f"PRAGMA table_info({table_name})")
            existing_columns = {row[1] for row in cur.fetchall()}

            for column in table.columns:
                if column.name in existing_columns:
                    continue
                # 构造 ADD COLUMN 语句（只取列名 + SQLite 类型）
                col_type = column.type.compile(dialect=sqlite.dialect())
                col_spec = f"{column.name} {col_type}"
                if not column.nullable:
                    col_spec += " NOT NULL"
                sql = f"ALTER TABLE {table_name} ADD COLUMN {col_spec}"
                try:
                    cur.execute(sql)
                except sqlite3.OperationalError as e:
                    # 已存在或其他不支持的变更，记录并继续
                    print(f"[DB MIGRATE] 跳过 {table_name}.{column.name}: {e}")
        conn.commit()
    finally:
        conn.close()


async def get_db() -> AsyncSession:
    """获取数据库会话（FastAPI 依赖）。"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """初始化数据库表，并对 SQLite 自动补齐缺失列。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    if settings.database_url.startswith("sqlite"):
        await asyncio.to_thread(_migrate_sqlite, settings.database_url)
