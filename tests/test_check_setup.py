"""Tests for the setup checker (check_setup.py)."""

from check_setup import check_data, check_env, check_index


def test_check_env_missing_file(tmp_path):
    result = check_env(tmp_path / ".env")
    assert not result["ok"]
    assert "cp .env.example .env" in result["fix"]


def test_check_env_still_has_the_placeholder_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=your-groq-key\n")
    result = check_env(env)
    assert not result["ok"]
    assert "console.groq.com" in result["fix"]


def test_check_env_with_a_real_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=abc123\nOPENAI_API_BASE=https://api.groq.com/openai/v1\n")
    result = check_env(env)
    assert result["ok"]
    # the key itself must never be printed
    assert "abc123" not in result["detail"]


def test_check_data_counts_csv_files(tmp_path):
    assert not check_data(tmp_path)["ok"]
    for i in range(12):
        (tmp_path / f"{i}_Source_QA.csv").write_text("question,answer\n")
    result = check_data(tmp_path)
    assert result["ok"] and "12" in result["detail"]


def test_check_data_with_some_files_missing(tmp_path):
    for i in range(3):
        (tmp_path / f"{i}_Source_QA.csv").write_text("question,answer\n")
    result = check_data(tmp_path)
    assert not result["ok"]
    assert result["fix"] == "python download_data.py"


def test_check_index_without_the_folder(tmp_path):
    result = check_index(tmp_path / "chroma_db")
    assert not result["ok"]
    assert "build_index.py" in result["fix"]


def test_check_index_with_an_empty_folder(tmp_path):
    result = check_index(tmp_path)
    assert not result["ok"]
    assert "build_index.py" in result["fix"]
