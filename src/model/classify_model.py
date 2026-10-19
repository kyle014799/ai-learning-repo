"""文本情感分类推理模块。
使用 transformers pipeline 加载配置中的情感分类模型。
采用懒加载：首次请求才加载模型，避免服务启动阶段因网络问题直接崩溃。
"""
from __future__ import annotations

from typing import TypedDict

from transformers import pipeline

from ..config import settings
from ..utils.logger import logger

# distilbert SST-2 英文标签 -> 中文
_LABEL_MAP = {
    "positive": "正面",
    "negative": "负面",
    "label_1": "正面",
    "label_0": "负面",
    "positive (stars 4 and 5)": "正面",
    "negative (stars 1, 2 and 3)": "负面",
}

class ClassifyResult(TypedDict):
    """情感分类结果。"""
    label: str
    score: float

# 全局变量，初始为None，懒加载
_classifier: pipeline | None = None

def _load_classifier():
    """懒加载分类pipeline，仅首次调用执行"""
    global _classifier
    if _classifier is not None:
        return _classifier
    model_name = settings.classify_model_name
    logger.info("正在加载情感分类模型: {}", model_name)
    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
    )
    logger.info("情感分类模型加载完成: {}", model_name)
    _classifier = classifier
    return _classifier

def _to_zh_label(raw_label: str) -> str:
    """将模型原始标签映射为正面/负面。"""
    raw = raw_label.strip()
    # 优先完整匹配；找不到就拿开头单词匹配，兜底
    if raw in _LABEL_MAP:
        return _LABEL_MAP[raw]
    prefix = raw.split(" ")[0].lower()
    return _LABEL_MAP.get(prefix, raw)

def text_classify(text: str) -> ClassifyResult:
    """对输入文本做情感分类。
    Args:
        text: 待分类的原始字符串。
    Returns:
        包含 ``label`` 与 ``score`` 的结果。
        ``label`` 为「正面」「负面」；置信度低于阈值/模型加载失败时为「不确定」。
    """
    if not isinstance(text, str):
        text = str(text)
    cleaned = text.strip()
    if len(cleaned) == 0:
        logger.info("输入文本为空")
        return ClassifyResult(label="不确定", score=0.0)
    if len(cleaned) < settings.classify_min_text_length:
        logger.info(
            "文本过短，跳过分类: 长度={}, 阈值={}",
            len(cleaned),
            settings.classify_min_text_length,
        )
        return ClassifyResult(label="不确定", score=0.0)

    # 懒加载，第一次请求才加载模型
    try:
        classifier = _load_classifier()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("情感分类模型加载失败: {}", exc)
        return ClassifyResult(label="不确定", score=0.0)

    try:
        outputs = classifier(cleaned, truncation=True)
        item = outputs[0]
        raw_label = str(item["label"])
        score = float(item["score"])
    except (RuntimeError, ValueError, OSError, KeyError, IndexError, TypeError) as exc:
        logger.error("情感分类推理失败: {}", exc)
        return ClassifyResult(label="不确定", score=0.0)

    label = _to_zh_label(raw_label)
    if score < settings.classify_confidence_threshold:
        logger.info(
            "置信度低于阈值，返回不确定: score={}, threshold={}",
            score,
            settings.classify_confidence_threshold,
        )
        return ClassifyResult(label="不确定", score=score)
    return ClassifyResult(label=label, score=score)
