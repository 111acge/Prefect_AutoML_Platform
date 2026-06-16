#!/usr/bin/env python3
"""环境检查脚本：验证 Prefect AutoML Platform 的运行环境。"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠️  {msg}")


def err(msg: str) -> None:
    print(f"  ❌ {msg}")


def check_python_version() -> bool:
    print("检查 Python 版本...")
    major, minor = sys.version_info[:2]
    if major == 3 and 12 <= minor <= 13:
        ok(f"Python {major}.{minor}")
        return True
    err(f"Python {major}.{minor} 不符合要求（需要 3.12/3.13）")
    return False


def check_venv(project_root: Path) -> bool:
    print("检查虚拟环境...")
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        ok(f"虚拟环境存在: {venv_python}")
        return True
    err("虚拟环境不存在，请运行 uv venv 或 python -m venv .venv")
    return False


def check_module(module_name: str, optional: bool = False) -> bool:
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        ok(f"模块已安装: {module_name}")
        return True
    msg = f"模块未安装: {module_name}"
    if optional:
        warn(msg + "（可选）")
        return True
    err(msg)
    return False


def check_command(name: str, cmd: list[str]) -> bool:
    print(f"检查命令: {name} ...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        ok(f"{name} 可用")
        return True
    except Exception as e:
        err(f"{name} 不可用: {e}")
        return False


def check_ports() -> bool:
    print("检查服务端口占用...")
    ports = [8001, 8084, 4200]
    all_free = True
    for port in ports:
        try:
            result = subprocess.run(
                ["ss", "-tlnp", f"sport = :{port}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if f":{port}" in result.stdout:
                ok(f"端口 {port} 已被占用（服务可能正在运行）")
            else:
                warn(f"端口 {port} 未被占用")
        except Exception as e:
            warn(f"无法检查端口 {port}: {e}")
    return all_free


def check_env_vars() -> bool:
    print("检查环境变量...")
    db_url = os.environ.get("DATABASE_URL", "")
    if db_url:
        ok(f"DATABASE_URL 已设置")
    else:
        warn("DATABASE_URL 未设置，将使用默认 SQLite")
    return True


def main() -> int:
    project_root = Path(__file__).parent.parent.resolve()
    print(f"项目根目录: {project_root}\n")

    checks = [
        check_python_version(),
        check_venv(project_root),
        check_module("fastapi"),
        check_module("prefect"),
        check_module("autogluon"),
        check_module("pandas"),
        check_module("sklearn"),
        check_module("openai", optional=True),
        check_command("uv", ["uv", "--version"]),
        check_command("node", ["node", "--version"]),
        check_ports(),
        check_env_vars(),
    ]

    print("\n" + "=" * 40)
    if all(checks):
        print("✅ 环境检查全部通过")
        return 0
    else:
        print("❌ 环境检查存在失败项，请修复后再启动")
        return 1


if __name__ == "__main__":
    sys.exit(main())
