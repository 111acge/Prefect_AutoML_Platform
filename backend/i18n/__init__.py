# Copyright (C) 2026  Ethan FAN <fyz.214037@foxmail.com>
# This file is part of Prefect AutoML Platform and is licensed under AGPL-3.0-or-later.
# See LICENSE for details.

"""国际化模块入口。"""

from .core import I18n

# 全局单例
from config import settings

i18n = I18n(default_locale=settings.default_locale)

# 便捷导出
_ = gettext = i18n.gettext
set_locale = i18n.set_locale
get_locale = i18n.get_locale
resolve_locale = i18n.resolve_locale
supported_locales = i18n.supported_locales

__all__ = ["i18n", "_", "gettext", "set_locale", "get_locale", "resolve_locale", "supported_locales"]
