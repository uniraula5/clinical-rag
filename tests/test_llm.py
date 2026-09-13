"""Tests for the LLM connection settings (llm.py). No real API calls are made."""

import json

import pytest

import llm


class FakeOpenAI:
    """Records how the client was built instead of connecting to anything."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(llm, "_client", None)  # ignore any client built by an earlier test
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://example.test/v1")
    return llm.get_client()


def test_client_has_a_timeout_so_the_demo_cannot_hang(fake_client):
    # without this, one hung request blocks for 10 minutes before retrying
    assert fake_client.kwargs["timeout"] == llm.REQUEST_TIMEOUT_SECONDS
    assert llm.REQUEST_TIMEOUT_SECONDS <= 60


def test_client_retries_rate_limits(fake_client):
    assert fake_client.kwargs["max_retries"] == llm.MAX_RETRIES


def test_client_is_built_once_and_reused(fake_client):
    assert llm.get_client() is fake_client


def test_missing_key_gives_a_helpful_error(monkeypatch):
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match=".env"):
        llm.get_client()


def test_chat_json_recovers_when_the_model_adds_extra_text(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda system, user, json_mode: 'Sure!\n{"results": [1, 2]}\nHope that helps')
    assert llm.chat_json("rules", "question") == {"results": [1, 2]}


def test_chat_json_raises_when_there_is_no_json(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda system, user, json_mode: "no json here")
    with pytest.raises(json.JSONDecodeError):
        llm.chat_json("rules", "question")
