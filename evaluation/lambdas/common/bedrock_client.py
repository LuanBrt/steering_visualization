import re

import boto3
from botocore.config import Config

_RESULT_DIGIT_PATTERN = re.compile(r"Result:\s*([01])")


def build_client():
    return boto3.client(
        "bedrock-runtime",
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
    )


def converse_with_image(client, model_id: str, image_bytes: bytes, image_format: str, prompt_text: str, max_tokens: int) -> str:
    response = client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": {"format": image_format, "source": {"bytes": image_bytes}}},
                    {"text": prompt_text},
                ],
            }
        ],
        inferenceConfig={"maxTokens": max_tokens},
    )
    return response["output"]["message"]["content"][0]["text"].strip()


def converse_text(client, model_id: str, prompt_text: str, max_tokens: int) -> str:
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt_text}]}],
        inferenceConfig={"maxTokens": max_tokens},
    )
    return response["output"]["message"]["content"][0]["text"].strip()


def extract_result_digit(text: str) -> str:
    match = _RESULT_DIGIT_PATTERN.search(text)
    if not match:
        raise ValueError(f"no Result: 0/1 found in judge output: {text!r}")
    return match.group(1)
