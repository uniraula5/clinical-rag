"""
Small wrapper around the LLM API. Groq's API is OpenAI-compatible, so the
official openai package works once it's pointed at Groq's URL.

Settings come from .env (copy .env.example):
    OPENAI_API_KEY     your Groq API key
    OPENAI_API_BASE    https://api.groq.com/openai/v1
    OPENAI_MODEL_NAME  e.g. openai/gpt-oss-120b
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No API key found. Copy .env.example to .env and add your Groq key.")
        # extra retries help with the free tier's rate limits
        _client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_API_BASE"), max_retries=5)
    return _client


def model_name():
    return os.getenv("OPENAI_MODEL_NAME", "openai/gpt-oss-120b")


def chat(system, user, json_mode=False):
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = get_client().chat.completions.create(
        model=model_name(),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,  # most repeatable output, which makes testing easier
        **extra,
    )
    return response.choices[0].message.content.strip()


def chat_json(system, user):
    text = chat(system, user, json_mode=True)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fallback: pull out the first {...} block if the model added extra text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            raise
        return json.loads(match.group(0))
