"""
应用管线模块

本模块是 StatAgent 的核心应用入口。
核心类：
- EvidenceLoop: 证据循环，协调整个推断流程
- ExecutionInput: 执行输入，包含数据和参数
- StatAgentConfig: 应用配置
- build_default_app: 构建默认应用实例

设计原则：
- 证据循环从用户输入到报告生成，完整编排推断流程
- 支持 Web 界面 (Streamlit) 和命令行两种交互模式
"""

from statagent.app.pipeline import EvidenceLoop, ExecutionInput, StatAgentConfig, build_default_app

__all__ = ["EvidenceLoop", "ExecutionInput", "StatAgentConfig", "build_default_app"]