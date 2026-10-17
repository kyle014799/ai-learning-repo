"""summary_model.summary_infer 的 pytest 单元测试。

只覆盖模型推理函数本身，通过 mock pipeline 避免加载真实模型和启动 Web 服务。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# 将 src 目录加入模块搜索路径，保证测试能导入 model.summary_model
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 单元测试不加载真实 transformers/pipeline，导入前先注入占位模块
sys.modules.setdefault("transformers", MagicMock())

from model.summary_model import MIN_TEXT_LENGTH, summary_infer

# 过短文本时的固定提示，与实现保持一致
TOO_SHORT_MSG = "文本过短，无法生成摘要"

# 场景 1 使用的正常长文本（长度必须 >= MIN_TEXT_LENGTH）
NORMAL_LONG_TEXT = (
    "Natural language processing enables computers to understand and generate "
    "human language. Summarization models compress long documents into a short "
    "overview while preserving the key facts."
)


@patch("model.summary_model._get_summarizer")
def test_summary_infer_normal_long_text(mock_get_summarizer):
    """场景 1：正常长文本，成功返回摘要字符串。

    文本长度达到阈值后会调用摘要 pipeline；此处 mock 返回值，
    断言函数会取出 summary_text 并 strip 后返回。
    """
    expected_summary = "NLP models compress documents into short overviews."
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"summary_text": f"  {expected_summary}  "}]
    mock_get_summarizer.return_value = mock_pipeline

    result = summary_infer(NORMAL_LONG_TEXT)

    assert isinstance(result, str)
    assert result == expected_summary
    mock_get_summarizer.assert_called_once()
    mock_pipeline.assert_called_once()
    # 传入 pipeline 的应是 strip 后的原文
    call_args, call_kwargs = mock_pipeline.call_args
    assert call_args[0] == NORMAL_LONG_TEXT.strip()
    assert call_kwargs["truncation"] is True


@patch("model.summary_model._get_summarizer")
def test_summary_infer_empty_string(mock_get_summarizer):
    """场景 2：输入空字符串，返回提示“文本过短，无法生成摘要”。

    空串和仅空白字符串在 strip 后长度都为 0，应直接返回提示，不调用模型。
    """
    assert summary_infer("") == TOO_SHORT_MSG
    assert summary_infer("   \n\t  ") == TOO_SHORT_MSG
    mock_get_summarizer.assert_not_called()


@patch("model.summary_model._get_summarizer")
def test_summary_infer_very_short_text(mock_get_summarizer):
    """场景 3：输入很短的文本（少于 50 字符），返回提示信息。

    非空但低于 MIN_TEXT_LENGTH 的文本同样不调用模型，返回过短提示。
    """
    short_text = "这是一段很短的文本。"
    assert len(short_text) < MIN_TEXT_LENGTH

    result = summary_infer(short_text)

    assert result == TOO_SHORT_MSG
    mock_get_summarizer.assert_not_called()
