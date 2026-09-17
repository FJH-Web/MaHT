"""
统计方法库模块

本模块提供预部署的本地统计算法，不执行任何生成的代码。

当前版本只包含假设检验方法，不再安装或暴露：
- 机器学习算法
- 回归分析
- 降维方法
- 纯描述统计

每个方法都有对应的 Method Card，包含方法描述、输入输出、假设条件等。
新增方法只需注册新的 Card 与预部署函数，无需修改路由器。
"""

from statagent.statistical_library.catalog import build_default_registry

__all__ = ["build_default_registry"]