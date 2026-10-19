"""文本情感分类接口单元测试"""
import pytest
from src.model.classify_model import text_classify

def test_classify_positive():
    """测试正面评价文本"""
    res = text_classify("这家店味道很好，上菜速度快，非常满意！")
    assert res["label"] == "正面"
    assert res["score"] > 0.7

def test_classify_negative():
    """测试负面评价文本"""
    res = text_classify("很难吃，上菜很慢，体验很差。")
    assert res["label"] == "负面"
    assert res["score"] > 0.7

def test_text_too_short():
    """测试过短文本，返回不确定"""
    res = text_classify("不错")
    assert res["label"] == "不确定"

def test_empty_text_direct_call():
    """直接调用函数传入空文本，不走http接口，返回不确定"""
    res = text_classify("")
    assert res["label"] == "不确定"
    assert res["score"] == 0.0


# 放在test_classify.py文件最末尾
if __name__ == "__main__":
    res = text_classify("这家店味道很好，上菜速度快，非常满意！")
    print(res)
    assert res["label"] == "正面"
    print("✅测试通过")
