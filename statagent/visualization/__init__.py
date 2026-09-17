"""
可视化模块

本模块提供统计可视化和中文显示支持。
核心组件：
- VisualizationEngine: 方法专属结果可视化引擎
- ExploratoryVisualizationEngine: 探索性数据分析可视化引擎
- configure_chinese_font: Matplotlib 中文字体配置

设计原则：
- 图表在本地生成，确保可复现性
- 支持中文标签和标题
- 每张 Method Card 声明方法专属可视化配方
"""

from statagent.visualization.engine import VisualizationEngine
from statagent.visualization.eda import ExploratoryVisualizationEngine
from statagent.visualization.fonts import configure_chinese_font

__all__ = ["ExploratoryVisualizationEngine", "VisualizationEngine", "configure_chinese_font"]