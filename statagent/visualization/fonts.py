"""Matplotlib 中文字体配置，兼容 Windows、macOS 与常见 Linux 环境。"""

from __future__ import annotations

import matplotlib
from matplotlib import font_manager


def configure_chinese_font() -> str:
    candidates = (
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in available), "DejaVu Sans")
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 120
    return selected

