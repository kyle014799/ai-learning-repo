"""文本关键词提取接口。"""

import re
from collections import Counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Keyword Extraction API")

# 返回词频最高的关键词数量
TOP_K = 3

# 常见中英文停用词，分词后会被过滤
STOPWORDS = {
    "的",
    "了",
    "在",
    "是",
    "我",
    "有",
    "和",
    "就",
    "不",
    "人",
    "都",
    "一",
    "一个",
    "上",
    "也",
    "很",
    "到",
    "说",
    "要",
    "去",
    "你",
    "会",
    "着",
    "没有",
    "看",
    "好",
    "自己",
    "这",
    "他",
    "她",
    "它",
    "们",
    "与",
    "及",
    "或",
    "但",
    "把",
    "被",
    "从",
    "对",
    "而",
    "于",
    "以",
    "为",
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "this",
    "that",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
}


class ExtractRequest(BaseModel):
    """关键词提取请求体。"""

    text: str = Field(..., description="待提取关键词的输入文本")


class ExtractResponse(BaseModel):
    """关键词提取响应体。"""

    keywords: list[str]


def tokenize(text: str) -> list[str]:
    """对文本进行简单分词。

    英文按单词切分，中文按连续汉字切分为二字词（单字单独保留）。
    """
    raw_tokens = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]+", text.lower())
    words: list[str] = []
    for token in raw_tokens:
        if re.fullmatch(r"[a-z]+", token):
            words.append(token)
            continue
        if len(token) == 1:
            words.append(token)
            continue
        for index in range(len(token) - 1):
            words.append(token[index : index + 2])
    return words


def extract_top_keywords(text: str, top_k: int = TOP_K) -> list[str]:
    """过滤停用词后，返回出现频率最高的 top_k 个词。"""
    words = [word for word in tokenize(text) if word not in STOPWORDS]
    if not words:
        return []
    counter = Counter(words)
    return [word for word, _ in counter.most_common(top_k)]


@app.post("/extract_keyword", response_model=ExtractResponse)
def extract_keyword(payload: ExtractRequest) -> ExtractResponse:
    """提取文本中出现频率最高的关键词。"""
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="文本不能为空")
    keywords = extract_top_keywords(text)
    return ExtractResponse(keywords=keywords)
