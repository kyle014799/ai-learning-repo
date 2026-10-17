"""文本摘要模型推理模块。

使用 transformers pipeline 加载 facebook/bart-base，
模块内全局只初始化一次，避免每次请求重复加载。
"""
import os

# 配置国内镜像，解决国内无法访问huggingface.co下载模型的问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import pipeline

# 低于该字符数视为过短，不调用模型
MIN_TEXT_LENGTH = 50

# 摘要模型名称
MODEL_NAME = "facebook/bart-base"

# 全局 pipeline，进程内只创建一次
_summarizer = None


def _get_summarizer():
    """获取全局摘要 pipeline，首次调用时加载模型。"""
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline(
            "summarization",
            model=MODEL_NAME,
        )
    return _summarizer


def summary_infer(text: str) -> str:
    """对输入文本生成摘要。

    Args:
        text: 待摘要的原始字符串。

    Returns:
        摘要文本；文本过短时返回提示；推理失败时返回异常信息。
    """
    if not isinstance(text, str):
        text = str(text)

    cleaned = text.strip()
    if len(cleaned) < MIN_TEXT_LENGTH:
        return "文本过短，无法生成摘要"

    try:
        summarizer = _get_summarizer()
        outputs = summarizer(
            cleaned,
            max_length=128,
            min_length=16,
            do_sample=False,
            truncation=True,
        )
        return outputs[0]["summary_text"].strip()
    except (RuntimeError, ValueError) as exc:
        return f"摘要推理失败: {exc}"
