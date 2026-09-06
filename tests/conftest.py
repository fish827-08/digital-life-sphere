"""pytest 根路径配置。

让 tests/ 下的用例能 import 项目的顶层包（core / simulation / world）。
"""
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)