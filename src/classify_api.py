"""文本情感分类 FastAPI 接口。

基于 ``src/model/classify_model.py`` 中的 ``text_classify``，
对外提供 POST ``/classify`` 接口。
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from .model.classify_model import text_classify
from .utils.logger import logger

app = FastAPI(title="Text Classify API")


class ClassifyRequest(BaseModel):
    """分类请求体。"""

    text: str = Field(..., min_length=1, description="待分类的输入文本")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """拒绝空字符串以及仅包含空白的文本。"""
        if not value.strip():
            raise ValueError("文本不能为空")
        return value


class ClassifyResponse(BaseModel):
    """分类响应体：输入文本、标签、置信度。"""

    input_text: str
    label: str
    score: float = Field(..., description="置信度分数")


def _error_body(code: int, message: str, detail: Any = None) -> dict[str, Any]:
    """构造统一 JSON 错误体。"""
    return {"code": code, "message": message, "detail": detail}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """请求体校验失败时返回统一错误 JSON，并写入日志。"""
    logger.error(
        "请求参数校验失败: path={}, errors={}",
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(
        status_code=422,
        content=_error_body(422, "请求参数校验失败", exc.errors()),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """HTTP 异常返回统一错误 JSON，并写入日志。"""
    logger.error(
        "HTTP 异常: path={}, status={}, detail={}",
        request.url.path,
        exc.status_code,
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(exc.status_code, "请求处理失败", exc.detail),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """捕获其余接口异常，返回统一错误 JSON，并写入日志。"""
    logger.error("接口异常: path={}, error={}", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content=_error_body(500, "服务器内部错误", str(exc)),
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify_text(payload: ClassifyRequest) -> ClassifyResponse:
    """根据输入文本进行情感分类。

    调用模型层 ``text_classify``，记录输入文本与推理耗时。
    """
    input_text = payload.text
    started = time.perf_counter()
    result = text_classify(input_text)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "分类请求完成: input_text={}, elapsed_ms={:.2f}",
        input_text,
        elapsed_ms,
    )
    return ClassifyResponse(
        input_text=input_text,
        label=result["label"],
        score=result["score"],
    )
