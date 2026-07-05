"""Kor-Smishing 데이터 로딩과 평가용 샘플 구성.

출처: https://github.com/Ez-Sy01/KOR_phishing_Detect-Dataset
class 0 = ham(정상) 42,594 / class 1 = smishing 615 (총 43,209)
원본은 data/raw/ (gitignore). 파생 샘플도 커밋하지 않는다(라이선스 미명시).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR

KOR_SMISHING_CSV = RAW_DIR / "KOR_phishing_data_raw.csv"
EVAL_SAMPLE_CSV = PROCESSED_DIR / "eval_sample.csv"


def load_kor_smishing(path: Path = KOR_SMISHING_CSV) -> pd.DataFrame:
    """content, class 컬럼을 가진 DataFrame 반환."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음. GitHub 저장소에서 KOR_phishing_data_raw.zip 을 받아 압축을 푸세요."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["content"]).copy()
    df["content"] = df["content"].astype(str).str.strip()
    df = df[df["content"] != ""]
    df["class"] = df["class"].astype(int)
    return df.reset_index(drop=True)


def make_eval_sample(
    n_per_class: int = 100, seed: int = 42, save: bool = True
) -> pd.DataFrame:
    """클래스 균형을 맞춘 결정론적 평가 샘플을 생성/저장.

    train/serve 누수 방지를 위해 원본 index 를 보존한다(향후 학습에서 제외용).
    """
    df = load_kor_smishing()
    smish = df[df["class"] == 1]
    ham = df[df["class"] == 0]
    n_s = min(n_per_class, len(smish))
    n_h = min(n_per_class, len(ham))
    sample = pd.concat(
        [
            smish.sample(n=n_s, random_state=seed),
            ham.sample(n=n_h, random_state=seed),
        ]
    ).sample(frac=1.0, random_state=seed)  # shuffle
    sample = sample.reset_index().rename(columns={"index": "orig_index"})
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        sample.to_csv(EVAL_SAMPLE_CSV, index=False, encoding="utf-8-sig")
    return sample


def get_eval_sample(n_per_class: int = 100, seed: int = 42) -> pd.DataFrame:
    """저장된 평가 샘플이 있으면 로드, 없으면 생성."""
    if EVAL_SAMPLE_CSV.exists():
        return pd.read_csv(EVAL_SAMPLE_CSV, encoding="utf-8-sig")
    return make_eval_sample(n_per_class=n_per_class, seed=seed)
