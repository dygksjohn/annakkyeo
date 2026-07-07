"""증강 — 합성 스미싱 생성 (개발 PC, GPT API).

목적: 소수 클래스(스미싱) 확대 + EDA에서 드러난 편향/구식화 보정.
유형 축(사칭대상 × 수법 × 말투)으로 다양성을 확보하고, 각 문자에 판정+설명 라벨을
함께 생성해 SFT 레코드로 바로 저장한다.

EDA 근거(docs/EDA_Kor-Smishing.md): 스미싱의 73.7%는 URL 없음, app_install 1.6%·택배 12%로
과소 → 수법에 '회신전화/송금요구/APK설치', 대상에 '택배' 비중을 명시적으로 포함.

개인정보 금지: 실명·실제 번호·실제 URL 대신 그럴듯한 가짜 값 사용.
출력: data/processed/sft_aug.jsonl (chat 포맷, gitignore)

실행: python -m src.augment --n 800          # 본 증강
      python -m src.augment --limit 12       # 소량 검증
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .build_sft_data import to_chat_record
from .config import BASELINE_MODEL, PROCESSED_DIR, get_openai_client
from .schema import RISK_FACTORS, parse_model_output

# 유형 축 (EDA 반영: 택배·회신전화·송금요구 비중↑)
TARGETS = [
    "택배·물류 사칭", "택배·물류 사칭",  # 가중치↑(EDA에서 과소)
    "정부·수사기관 사칭", "건강보험·질병관리청 사칭", "법원·검찰 사칭",
    "은행·카드사 사칭", "증권·투자 유인",
    "지인·가족 사칭", "부고·청첩장 사칭", "국세청 환급 사칭",
]
METHODS = [
    "단축 URL 클릭 유도", "APK 앱 설치 유도", "APK 앱 설치 유도",  # app_install 보강
    "회신 전화 유도(URL 없음)", "회신 전화 유도(URL 없음)",         # URL 없는 유형 보강
    "송금·계좌이체 요구(URL 없음)", "개인정보·인증번호 요구",
]
TONES = ["공식 안내체", "다급·압박체", "친근한 말투"]

AUG_SYSTEM = (
    "당신은 한국어 스미싱 탐지 모델의 학습 데이터를 만드는 전문가입니다.\n"
    "실제로 유포될 법한 현실적인 스미싱 문자 1건을 생성하고, 그 판정 라벨을 함께 출력하세요.\n"
    "규칙:\n"
    "- 개인정보 금지: 실명·실제 전화번호·실제 URL 대신 그럴듯한 가짜 값 사용"
    "(예: http://bit.ly/x4k2, 010-0000-0000 형태 피하고 '지정번호'로 표현).\n"
    "- risk_factors 는 아래 태그에서만 선택:\n"
    f"  {', '.join(sorted(RISK_FACTORS))}\n"
    "- explanation 은 2~3문장으로 왜 위험한지 구체적으로.\n"
    "반드시 아래 JSON 형식으로만 답하세요.\n"
    '{"message": "문자원문", "verdict": "smishing", "risk_factors": [...], "explanation": "..."}'
)


def build_aug_messages(target: str, method: str, tone: str) -> list[dict]:
    user = (
        f"사칭 대상: {target}\n수법: {method}\n말투: {tone}\n\n"
        "위 조합으로 현실적인 한국어 스미싱 문자와 라벨을 생성하세요."
    )
    return [{"role": "system", "content": AUG_SYSTEM}, {"role": "user", "content": user}]


def gen_one(client, combo, max_retries: int = 3) -> dict | None:
    target, method, tone = combo
    messages = build_aug_messages(target, method, tone)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=BASELINE_MODEL, messages=messages, temperature=0.9,
                response_format={"type": "json_object"},
            )
            obj = json.loads(resp.choices[0].message.content or "{}")
            msg = str(obj.get("message", "")).strip()
            if not msg:
                return None
            p = parse_model_output(json.dumps(obj, ensure_ascii=False))
            rf = [t for t in p.risk_factors if t in RISK_FACTORS]
            return {"message": msg, "verdict": "smishing", "risk_factors": rf,
                    "explanation": p.explanation, "combo": f"{target}|{method}|{tone}"}
        except Exception:  # noqa: BLE001
            if attempt == max_retries - 1:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def run(n: int, limit: int | None, workers: int = 8, seed: int = 42) -> None:
    client = get_openai_client()
    rng = random.Random(seed)
    total = limit or n
    combos = [(rng.choice(TARGETS), rng.choice(METHODS), rng.choice(TONES)) for _ in range(total)]

    results: list[dict | None] = [None] * len(combos)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(gen_one, client, c): i for i, c in enumerate(combos)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="증강생성"):
            results[futs[fut]] = fut.result()

    # 중복(정규화 후 완전일치) 제거
    seen: set[str] = set()
    recs, dropped = [], 0
    for r in results:
        if r is None:
            continue
        key = "".join(r["message"].split())
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        recs.append(to_chat_record(r["message"], r))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "sft_aug.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n생성 {len(recs)}건 저장 (중복 {dropped} 제거) → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="합성 스미싱 증강")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--limit", type=int, default=None, help="검증용 상한")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run(args.n, args.limit, args.workers)


if __name__ == "__main__":
    main()
