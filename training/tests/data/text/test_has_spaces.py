#!/usr/bin/env python3

from caiman_asr_train.data.text.has_spaces import split_into_logical_tokens


def test_split_works():
    s = split_into_logical_tokens
    assert s("I have 12 cats.") == ["I", "have", "12", "cats."]
    assert s("Tengo 12 gatos.") == ["Tengo", "12", "gatos."]
    assert s("لدي 12 قطة.") == ["لدي", "12", "قطة."]
    assert s("יש לי 12 חתולים.") == ["יש", "לי", "12", "חתולים."]
    assert s("我有 12 只猫。") == ["我", "有", "12", "只", "猫", "。"]
    assert s("Έχω 12 γάτες.") == ["Έχω", "12", "γάτες."]
    # Unsure if these are handled correctly:
    # In particular, Japanese hiragana and katakana
    # may need support.
    assert s("나는 고양이 12마리를 키우고 있어요.") == [
        "나는",
        "고양이",
        "12마리를",
        "키우고",
        "있어요.",
    ]
    assert s("私は猫を12匹飼っています。") == [
        "私",
        "は",
        "猫",
        "を12",
        "匹",
        "飼",
        "っています。",
    ]


def test_code_switching():
    # Example from https://huggingface.co/datasets/CAiRE/ASCEND
    assert split_into_logical_tokens("嗯初次见面nice to meet you嗯") == [
        "嗯",
        "初",
        "次",
        "见",
        "面",
        "nice",
        "to",
        "meet",
        "you",
        "嗯",
    ]
