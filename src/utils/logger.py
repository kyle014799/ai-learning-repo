"""全局日志配置。

基于 loguru，同时输出到控制台和 ``logs/app.log``。
文件按大小自动切割，项目内通过本模块导出的 ``logger`` 共用同一实例。
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

# 日志目录放在仓库根目录下，避免受当前工作目录影响
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = _PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"

# 时间、级别、模块名、信息
_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level:<8}</level> | "
    "<cyan>{name}</cyan> | "
    "{message}"
)
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name} | {message}"
)

# 单文件超过该大小后切割，避免无限膨胀
_ROTATION_SIZE = "10 MB"
# 最多保留的历史日志份数
_RETENTION_COUNT = 5
_CONFIGURED = False


def setup_logger() -> None:
    """配置控制台与文件两个 sink。重复导入时不会重复添加。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()

    logger.add(
        sys.stderr,
        format=_CONSOLE_FORMAT,
        level="INFO",
        colorize=True,
        enqueue=True,
    )
    logger.add(
        LOG_FILE,
        format=_FILE_FORMAT,
        level="INFO",
        encoding="utf-8",
        rotation=_ROTATION_SIZE,
        retention=_RETENTION_COUNT,
        enqueue=True,
    )
    _CONFIGURED = True


setup_logger()

__all__ = ["LOG_DIR", "LOG_FILE", "logger"]
