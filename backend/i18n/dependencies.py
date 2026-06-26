# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""FastAPI locale 依赖。"""

from fastapi import Request

from . import set_locale, resolve_locale


async def get_locale(request: Request) -> str:
    """从请求头解析并设置当前语言。"""
    accept_language = request.headers.get("accept-language")
    locale = resolve_locale(accept_language)
    set_locale(locale)
    return locale
