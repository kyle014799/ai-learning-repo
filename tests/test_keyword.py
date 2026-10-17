"""keyword_api 关键词提取接口的 pytest 单元测试。"""

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

# 将 src 目录加入模块搜索路径，保证测试能导入 keyword_api
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from keyword_api import ExtractRequest, extract_keyword


def test_extract_keywords_from_normal_text():
    """场景 1：正常文本提取关键词。

    重复出现的英文词应排在结果前列，且返回数量不超过 TOP_K（默认 3）。
    """
    text = "python python python java java rust rust golang"
    response = extract_keyword(ExtractRequest(text=text))

    assert response.keywords, "正常文本应返回至少一个关键词"
    assert len(response.keywords) <= 3
    # python 出现 3 次，词频最高，应排在第一位
    assert response.keywords[0] == "python"
    assert "java" in response.keywords
    assert "rust" in response.keywords


def test_extract_keywords_empty_string():
    """场景 2：输入空字符串（含仅空白）应返回 400。"""
    with pytest.raises(HTTPException) as exc_info:
        extract_keyword(ExtractRequest(text=""))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "文本不能为空"

    # 去掉两端空白后仍为空，同样视为无效输入
    with pytest.raises(HTTPException) as exc_info:
        extract_keyword(ExtractRequest(text="   \n\t  "))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "文本不能为空"


def test_extract_keywords_very_short_text():
    """场景 3：输入很短文本。

    非空短文本应正常处理：有效词会被提取，纯停用词则返回空列表。
    """
    # 单个英文单词会被小写后作为关键词返回
    response = extract_keyword(ExtractRequest(text="AI"))
    assert response.keywords == ["ai"]

    # 单个中文停用词会被过滤，结果为空列表而非报错
    response = extract_keyword(ExtractRequest(text="的"))
    assert response.keywords == []
    assert isinstance(response.keywords, list)
