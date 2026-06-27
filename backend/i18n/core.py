# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""轻量级国际化核心实现。"""

import json
import re
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional


class I18n:
    """基于 JSON 字典的轻量 i18n 实现。"""

    def __init__(self, default_locale: str = "zh-CN", locales_dir: Optional[Path] = None):
        self.default_locale = default_locale
        self._locale: ContextVar[str] = ContextVar("locale", default=default_locale)
        self._catalogs: Dict[str, Dict[str, Any]] = {}

        if locales_dir is None:
            locales_dir = Path(__file__).parent / "locales"
        self.locales_dir = Path(locales_dir)
        self._load_catalogs()

    def __getstate__(self) -> Dict[str, Any]:
        """支持 pickle：ContextVar 不可序列化，重建时重新初始化。"""
        return {
            "default_locale": self.default_locale,
            "locales_dir": str(self.locales_dir),
            "_catalogs": self._catalogs,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """从 pickle 恢复时重新创建 ContextVar。"""
        self.default_locale = state["default_locale"]
        self._locale = ContextVar("locale", default=self.default_locale)
        self._catalogs = state.get("_catalogs", {})
        self.locales_dir = Path(state["locales_dir"])

    def _load_catalogs(self) -> None:
        """加载所有语言包。"""
        if not self.locales_dir.exists():
            return
        for path in self.locales_dir.glob("*.json"):
            locale = path.stem
            try:
                with path.open("r", encoding="utf-8") as f:
                    self._catalogs[locale] = json.load(f)
            except Exception:
                # 忽略损坏的语言包
                continue

    @property
    def supported_locales(self):
        """返回支持的语言列表。"""
        return list(self._catalogs.keys()) or [self.default_locale]

    def set_locale(self, locale: str) -> None:
        """设置当前上下文语言。"""
        if locale in self._catalogs:
            self._locale.set(locale)
        else:
            self._locale.set(self.default_locale)

    def get_locale(self) -> str:
        """获取当前上下文语言。"""
        return self._locale.get(self.default_locale)

    def gettext(self, key: str, default: Optional[str] = None, **kwargs) -> str:
        """获取翻译文本，支持嵌套 key 和变量插值。

        Args:
            key: 翻译键，支持点号分隔嵌套，如 "dataset.not_found"。
            default: 缺失时的默认文本，默认返回 key。
            **kwargs: 插值变量。
        """
        locale = self.get_locale()
        catalog = self._catalogs.get(locale) or self._catalogs.get(self.default_locale) or {}

        value: Any = catalog
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                value = None
                break

        if value is None:
            # 尝试 fallback 到默认语言
            if locale != self.default_locale:
                value = self._catalogs.get(self.default_locale, {})
                for part in key.split("."):
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break

        if value is None:
            text = default if default is not None else key
        else:
            text = str(value)

        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
        return text

    def resolve_locale(self, accept_language: Optional[str]) -> str:
        """从 Accept-Language 头解析最匹配的语言。"""
        if not accept_language:
            return self.default_locale

        # 解析 "zh-CN,zh;q=0.9,en;q=0.8" -> [('zh-CN', 1.0), ('zh', 0.9), ('en', 0.8)]
        locales: list[tuple[str, float]] = []
        for item in accept_language.split(","):
            item = item.strip()
            if not item:
                continue
            if ";q=" in item:
                tag, q = item.split(";q=", 1)
                try:
                    locales.append((tag.strip(), float(q.strip())))
                except ValueError:
                    locales.append((item, 1.0))
            else:
                locales.append((item, 1.0))

        # 按质量排序
        locales.sort(key=lambda x: x[1], reverse=True)

        # 优先完全匹配，再尝试前缀匹配
        supported = set(self.supported_locales)
        for tag, _ in locales:
            if tag in supported:
                return tag
            # 尝试 "zh" -> "zh-CN"
            for supported_locale in supported:
                if supported_locale.lower().startswith(tag.lower() + "-") or supported_locale.lower() == tag.lower():
                    return supported_locale

        return self.default_locale
