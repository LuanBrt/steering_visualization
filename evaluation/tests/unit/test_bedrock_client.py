from unittest.mock import MagicMock

import pytest

from common.bedrock_client import converse_text, converse_with_image, extract_result_digit


def _converse_response(text: str) -> dict:
    return {"output": {"message": {"content": [{"text": text}]}}}


def test_converse_with_image_sends_image_and_text_blocks_and_returns_stripped_text():
    client = MagicMock()
    client.converse.return_value = _converse_response("  giraffe  ")

    result = converse_with_image(
        client,
        model_id="us.openai.gpt-5.6-terra",
        image_bytes=b"fake-bytes",
        image_format="png",
        prompt_text="What is in the image? One word.",
        max_tokens=20,
    )

    assert result == "giraffe"
    call_kwargs = client.converse.call_args.kwargs
    assert call_kwargs["modelId"] == "us.openai.gpt-5.6-terra"
    assert call_kwargs["inferenceConfig"] == {"maxTokens": 20}
    content = call_kwargs["messages"][0]["content"]
    assert content[0]["image"]["format"] == "png"
    assert content[0]["image"]["source"]["bytes"] == b"fake-bytes"
    assert content[1]["text"] == "What is in the image? One word."


def test_converse_text_sends_single_text_block():
    client = MagicMock()
    client.converse.return_value = _converse_response("Result: 1")

    result = converse_text(client, model_id="us.openai.gpt-5.6-luna", prompt_text="judge this", max_tokens=200)

    assert result == "Result: 1"
    call_kwargs = client.converse.call_args.kwargs
    assert call_kwargs["messages"][0]["content"] == [{"text": "judge this"}]
    assert call_kwargs["inferenceConfig"] == {"maxTokens": 200}


def test_extract_result_digit_finds_zero():
    assert extract_result_digit("some reasoning...\nResult: 0") == "0"


def test_extract_result_digit_finds_one_with_extra_whitespace():
    assert extract_result_digit("reasoning\nResult:   1  \n") == "1"


def test_extract_result_digit_raises_when_missing():
    with pytest.raises(ValueError):
        extract_result_digit("the model rambled without a verdict")
