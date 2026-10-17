"""文本摘要 FastAPI 接口。

基于 ``src/model/summary_model.py`` 中的 ``summary_infer``，
对外提供 POST ``/summary`` 接口。
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .model.summary_model import summary_infer

app = FastAPI(title="Text Summary API")


class SummaryRequest(BaseModel):
    """摘要请求体。"""

    text: str = Field(..., min_length=1, description="待摘要的输入文本")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """拒绝空字符串以及仅包含空白的文本。"""
        if not value.strip():
            raise ValueError("文本不能为空")
        return value


class SummaryResponse(BaseModel):
    """摘要响应体，对应自定义 JSON 结构。"""

    input_text: str
    summary: str


@app.post("/summary", response_model=SummaryResponse)
def create_summary(payload: SummaryRequest) -> SummaryResponse:
    """根据输入文本生成摘要。

    Raises:
        HTTPException: 模型推理过程中发生异常时返回 500。
    """
    input_text = payload.text
    try:
        summary = summary_infer(input_text)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"摘要推理失败: {exc}",
        ) from exc

    # summary_infer 内部也会吞掉异常并返回错误字符串
    if summary.startswith("摘要推理失败:"):
        raise HTTPException(status_code=500, detail=summary)

    return SummaryResponse(input_text=input_text, summary=summary)
