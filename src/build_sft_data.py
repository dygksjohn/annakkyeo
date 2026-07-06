"""SFT 학습 데이터 구축 — Kor-Smishing → 판정+설명 라벨(JSON) 형식.

이 프로젝트의 독자적 기여: 각 문자에 '왜 위험/안전한가' 설명 라벨을 GPT로 생성.
- 정답 verdict(class 0→normal, 1→smishing)를 조건으로 근거만 생성(label-conditioned)
- EXP-001 평가셋(eval_sample.csv)은 test 로 예약, 학습에서 제외(누수 방지)
- 출력: data/processed/sft_{split}.jsonl (chat 포맷, gitignore)

실행:
    python -m src.build_sft_data --limit 6      # 소량 검증
    python -m src.build_sft_data --train 250 --val 60   # 본 구축
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from .config import BASELINE_MODEL, PROCESSED_DIR, get_openai_client
from .data import EVAL_SAMPLE_CSV, load_kor_smishing
from .prompts import SYSTEM_PROMPT, build_label_gen_messages, build_user_prompt
from .schema import RISK_FACTORS, parse_model_output

LABEL_MODEL = BASELINE_MODEL  # 설명 라벨 생성 모델 (gpt-4o-mini 기본)
CLASS_TO_VERDICT = {0: "normal", 1: "smishing"}


def _excluded_indices() -> set[int]:
    """test 로 예약된 eval_sample 의 orig_index 집합 (학습 누수 방지)."""
    if EVAL_SAMPLE_CSV.exists():
        return set(pd.read_csv(EVAL_SAMPLE_CSV, encoding="utf-8-sig")["orig_index"])
    return set()


def build_split(n_train: int, n_val: int, seed: int = 42) -> dict[str, pd.DataFrame]:
    """test 예약분을 제외하고 클래스 균형 train/val 을 구성."""
    df = load_kor_smishing()
    df = df[~df.index.isin(_excluded_indices())]
    smish = df[df["class"] == 1]
    ham = df[df["class"] == 0]

    def take(pool, n, used):
        return pool[~pool.index.isin(used)].sample(n=min(n, len(pool)), random_state=seed)

    used: set[int] = set()
    out = {}
    for split, n in (("train", n_train), ("val", n_val)):
        s = take(smish, n // 2, used); used |= set(s.index)
        h = take(ham, n - len(s), used); used |= set(h.index)
        out[split] = pd.concat([s, h]).sample(frac=1.0, random_state=seed)
    return out


def gen_label(client, text: str, verdict: str, max_retries: int = 3) -> dict:
    """정답 verdict 를 조건으로 risk_factors+explanation 생성. 태그는 화이트리스트 필터."""
    messages = build_label_gen_messages(text, verdict)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=LABEL_MODEL, messages=messages, temperature=0.3,
                seed=42, response_format={"type": "json_object"},
            )
            p = parse_model_output(resp.choices[0].message.content or "")
            rf = [t for t in p.risk_factors if t in RISK_FACTORS]  # 미허용 태그 제거
            return {
                "verdict": verdict,  # 정답 강제(모델이 바꿔도 무시)
                "risk_factors": [] if verdict == "normal" else rf,
                "explanation": p.explanation,
                "parse_ok": p.parse_ok,
            }
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                return {"verdict": verdict, "risk_factors": [], "explanation": "", "parse_ok": False, "error": str(e)}
            time.sleep(2 * (attempt + 1))
    return {"verdict": verdict, "risk_factors": [], "explanation": "", "parse_ok": False}


def to_chat_record(text: str, target: dict) -> dict:
    """SFT 용 chat 포맷. assistant = 목표 JSON 문자열."""
    assistant = json.dumps(
        {"verdict": target["verdict"], "risk_factors": target["risk_factors"],
         "explanation": target["explanation"]},
        ensure_ascii=False,
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text)},
            {"role": "assistant", "content": assistant},
        ]
    }


def build(n_train: int, n_val: int, limit: int | None, workers: int = 6) -> None:
    client = get_openai_client()
    splits = build_split(n_train, n_val)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for split, sdf in splits.items():
        if limit:
            sdf = sdf.head(limit)
        rows = sdf.to_dict("records")
        results: list[dict | None] = [None] * len(rows)

        def work(i, row):
            verdict = CLASS_TO_VERDICT[int(row["class"])]
            return i, row["content"], gen_label(client, row["content"], verdict)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, i, r) for i, r in enumerate(rows)]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"라벨생성:{split}"):
                i, text, target = fut.result()
                results[i] = to_chat_record(text, target) if target["parse_ok"] else None

        recs = [r for r in results if r is not None]
        out = PROCESSED_DIR / f"sft_{split}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{split}] {len(recs)}/{len(rows)} 건 저장 → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT 학습 데이터(판정+설명) 구축")
    ap.add_argument("--train", type=int, default=250)
    ap.add_argument("--val", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None, help="split별 상한(검증용)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    build(args.train, args.val, args.limit, args.workers)


if __name__ == "__main__":
    main()
