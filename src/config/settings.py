"""项目运行时配置。

使用 pydantic-settings 从环境变量（及可选的 ``.env`` 文件）读取配置，
未设置时使用代码中的默认值。Hugging Face 国内镜像通过 ``HF_ENDPOINT`` 注入。
"""

from __future__ import annotations

import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置类。

    环境变量名与字段名对应（默认大写），例如 ``CLASSIFY_MODEL_NAME``。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    classify_model_name: str = Field(
        default="uer/roberta-base-finetuned-dianping-chinese",
        description="文本分类所用 Hugging Face 模型名称",
    )
    classify_confidence_threshold: float = Field(
        default=0.5,
        description="分类结果最低置信度阈值，低于该值视为不确定",
    )
    classify_min_text_length: int = Field(
        default=5,
        description="触发分类推理所需的最小文本长度（字符数）",
    )
    hf_endpoint: str = Field(
        default="https://hf-mirror.com",
        description="Hugging Face Hub 国内镜像地址，对应环境变量 HF_ENDPOINT",
    )

    @field_validator("classify_confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, value: float) -> float:
        """置信度必须落在 [0, 1] 区间。"""
        if not 0.0 <= value <= 1.0:
            raise ValueError("classify_confidence_threshold 必须在 0 到 1 之间")
        return value

    @field_validator("classify_min_text_length")
    @classmethod
    def validate_min_text_length(cls, value: int) -> int:
        """最小文本长度必须为正整数。"""
        if value <= 0:
            raise ValueError("classify_min_text_length 必须大于 0")
        return value

    @field_validator("hf_endpoint")
    @classmethod
    def validate_hf_endpoint(cls, value: str) -> str:
        """镜像地址不能为空。"""
        endpoint = value.strip()
        if not endpoint:
            raise ValueError("hf_endpoint 不能为空")
        return endpoint.rstrip("/")


def apply_hf_endpoint(endpoint: str) -> None:
    """将国内镜像写入进程环境变量，供 huggingface_hub / transformers 使用。"""
    os.environ["HF_ENDPOINT"] = endpoint


settings = Settings()
apply_hf_endpoint(settings.hf_endpoint)
