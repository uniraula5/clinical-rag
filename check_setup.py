"""
Checks that everything this project needs is in place, and tells you exactly
what to run if something is missing.

    python check_setup.py
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
EXPECTED_CSV_FILES = 12


def check_env(env_path=PROJECT_DIR / ".env"):
    """The Groq key is only needed for writing answers, not for search."""
    if not env_path.exists():
        return {"name": "API key (.env)", "ok": False,
                "detail": "no .env file",
                "fix": "cp .env.example .env, then paste your Groq key into it"}

    from dotenv import dotenv_values

    settings = dotenv_values(env_path)
    key = settings.get("OPENAI_API_KEY", "")
    if not key or key == "your-groq-key":
        return {"name": "API key (.env)", "ok": False,
                "detail": ".env has no real key yet",
                "fix": "get a free key at https://console.groq.com and put it in .env"}

    # without OPENAI_API_BASE the requests go to OpenAI instead of Groq, and a
    # Groq key is rejected there, which looks exactly like a bad key
    if not settings.get("OPENAI_API_BASE"):
        return {"name": "API key (.env)", "ok": False,
                "detail": "key found, but OPENAI_API_BASE is missing so requests would go to OpenAI",
                "fix": "add OPENAI_API_BASE=https://api.groq.com/openai/v1 to .env (see .env.example)"}
    return {"name": "API key (.env)", "ok": True, "detail": "key and Groq address found", "fix": ""}


def check_data(raw_dir=PROJECT_DIR / "data" / "raw"):
    found = len(list(raw_dir.glob("*.csv"))) if raw_dir.exists() else 0
    if found < EXPECTED_CSV_FILES:
        return {"name": "MedQuAD data", "ok": False,
                "detail": f"{found} of {EXPECTED_CSV_FILES} CSV files in data/raw/",
                "fix": "python download_data.py"}
    return {"name": "MedQuAD data", "ok": True, "detail": f"{found} CSV files", "fix": ""}


def check_index(chroma_dir=PROJECT_DIR / "chroma_db"):
    if not chroma_dir.exists():
        return {"name": "Search index", "ok": False, "detail": "no chroma_db/ folder",
                "fix": "python build_index.py  (takes about 4 minutes)"}

    import chromadb

    from build_index import COLLECTION_NAME

    try:
        count = chromadb.PersistentClient(path=str(chroma_dir)).get_collection(COLLECTION_NAME).count()
    except Exception:
        return {"name": "Search index", "ok": False, "detail": "chroma_db/ exists but has no collection",
                "fix": "python build_index.py"}
    if count == 0:
        return {"name": "Search index", "ok": False, "detail": "the index is empty",
                "fix": "python build_index.py"}
    return {"name": "Search index", "ok": True, "detail": f"{count:,} chunks", "fix": ""}


def run_checks():
    return [check_env(), check_data(), check_index()]


if __name__ == "__main__":
    print("Checking setup...\n")
    checks = run_checks()
    for check in checks:
        mark = "OK  " if check["ok"] else "MISSING"
        print(f"  [{mark}] {check['name']}: {check['detail']}")

    problems = [c for c in checks if not c["ok"]]
    if not problems:
        print("\nEverything is ready. Start the demo with:  python app.py")
    else:
        print("\nTo fix:")
        for problem in problems:
            print(f"  - {problem['name']}: {problem['fix']}")
        if all(p["name"] == "API key (.env)" for p in problems):
            print("\nSearch still works without an API key: python app.py (use the Search tab)")
