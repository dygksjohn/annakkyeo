"""환경 변수 로딩과 공용 경로/클라이언트 팩토리."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "outputs"))

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

BASELINE_MODEL = os.getenv("OPENAI_BASELINE_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")


@lru_cache(maxsize=1)
def get_openai_client():
    """OpenAI 클라이언트를 반환. OPENAI_API_KEY 가 없으면 명확한 오류."""
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY 가 설정되지 않았습니다. .env.example 을 .env 로 복사하고 값을 채우세요."
        )
    return OpenAI(api_key=key)
