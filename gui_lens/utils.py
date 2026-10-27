"""
gui_lens.utils — shared utilities.

Delegates JSON parsing to openprogram.functions.agentics.json_parsing
(single implementation, maintained in the OpenProgram project).
"""

try:  # openprogram 重构后 parse_json 迁到 _utils;保留旧路径兼容
    from openprogram.functions.agentics._utils import parse_json  # noqa: F401
except ImportError:  # pragma: no cover
    from openprogram.functions.agentics.json_parsing import parse_json  # noqa: F401
