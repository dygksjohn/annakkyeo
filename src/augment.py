"""증강 — 합성 스미싱/정상 생성 (개발 PC, GPT API).

--kind smishing : 소수 클래스(스미싱) 확대 + EDA 편향 보정 → sft_aug.jsonl
--kind normal   : 정상 하드네거티브(스미싱처럼 보이나 합법) → sft_aug_normal.jsonl
                  목적: 파인튜닝 모델의 정상 문자 과탐(FP) 감소 (EXP-004 근거)

각 문자에 판정+설명 라벨을 함께 생성해 SFT chat 레코드로 저장한다.
개인정보 금지: 실명·실제 번호·실제 URL 대신 그럴듯한 가짜 값 사용.
출력물은 gitignore(data/processed/). train_qlora 가 sft_aug*.jsonl 을 자동 포함.

실행: python -m src.augment --kind smishing --n 800
      python -m src.augment --kind normal   --n 800
      python -m src.augment --kind normal   --limit 12   # 소량 검증
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

# ── 스미싱 유형 축 (EDA 반영: 택배·회신전화·송금요구 비중↑) ──
TARGETS = [
    "택배·물류 사칭", "택배·물류 사칭",
    "정부·수사기관 사칭", "건강보험·질병관리청 사칭", "법원·검찰 사칭",
    "은행·카드사 사칭", "증권·투자 유인",
    "지인·가족 사칭", "부고·청첩장 사칭", "국세청 환급 사칭",
]
METHODS = [
    "단축 URL 클릭 유도", "APK 앱 설치 유도", "APK 앱 설치 유도",
    "회신 전화 유도(URL 없음)", "회신 전화 유도(URL 없음)",
    "송금·계좌이체 요구(URL 없음)", "개인정보·인증번호 요구",
]
TONES = ["공식 안내체", "다급·압박체", "친근한 말투"]

# ── 정상 하드네거티브 카테고리 (스미싱과 표면 특징이 겹치는 합법 문자) ──
NORMAL_CATEGORIES = [
    "실제 택배사 공식 배송 알림(공식 앱/도메인, 정상 운송장 조회)",
    "은행·카드사 정식 거래 알림(결제 승인/출금 내역)",
    "수신거부 번호가 명시된 합법 광고(실제 사업자)",
    "병원·식당·미용실 등 예약/방문 확인 안내",
    "본인이 방금 요청한 정상 인증번호(OTP) 안내",
    "공공기관 정식 안내(행정·재난·민원 처리 안내)",
    "지인·가족 간 일상 대화",
    "통신사·구독 서비스 정상 요금/이용 안내",
    "회사·학교의 정상 공지(일정·회의 안내)",
]

AUG_SMISHING_SYSTEM = (
    "당신은 한국어 스미싱 탐지 모델의 학습 데이터를 만드는 전문가입니다.\n"
    "실제로 유포될 법한 현실적인 스미싱 문자 1건을 생성하고, 그 판정 라벨을 함께 출력하세요.\n"
    "규칙:\n"
    "- 개인정보 금지: 실명·실제 전화번호·실제 URL 대신 그럴듯한 가짜 값 사용"
    "(예: http://bit.ly/x4k2, 번호는 '지정번호'로 표현).\n"
    "- risk_factors 는 아래 태그에서만 선택:\n"
    f"  {', '.join(sorted(RISK_FACTORS))}\n"
    "- explanation 은 2~3문장으로 왜 위험한지 구체적으로.\n"
    "반드시 아래 JSON 형식으로만 답하세요.\n"
    '{"message": "문자원문", "verdict": "smishing", "risk_factors": [...], "explanation": "..."}'
)

AUG_NORMAL_SYSTEM = (
    "당신은 한국어 스미싱 탐지 모델의 학습 데이터를 만드는 전문가입니다.\n"
    "**합법적이고 정상적인** 한국어 문자 1건을 생성하세요. 단, 스미싱과 표면적으로 닮은"
    " 요소(기관명·링크·긴급성·안내문)를 포함하되 **실제로는 안전한** 문자여야 합니다"
    "(하드네거티브). 목적은 모델이 정상 문자를 스미싱으로 오판하지 않도록 경계를 학습시키는 것.\n"
    "규칙:\n"
    "- 개인정보 금지: 실명·실제 번호·실제 URL 대신 그럴듯한 가짜 값 사용.\n"
    "- 합법성 신호를 자연스럽게 담기: 공식 도메인/앱, 수신거부 안내, 본인이 예상한 거래,"
    " 기만적 유도(앱 설치·개인정보 요구) 없음 중 해당되는 것.\n"
    "- verdict 는 normal, risk_factors 는 빈 배열 [].\n"
    "- explanation 은 2~3문장으로 '왜 안전한지 + 스미싱과 구별하는 일반 원칙'을 설명.\n"
    "반드시 아래 JSON 형식으로만 답하세요.\n"
    '{"message": "문자원문", "verdict": "normal", "risk_factors": [], "explanation": "..."}'
)


def _combo(kind: str, rng: random.Random):
    if kind == "smishing":
        return (rng.choice(TARGETS), rng.choice(METHODS), rng.choice(TONES))
    return (rng.choice(NORMAL_CATEGORIES), rng.choice(TONES))


def _messages(kind: str, combo) -> list[dict]:
    if kind == "smishing":
        target, method, tone = combo
        user = (f"사칭 대상: {target}\n수법: {method}\n말투: {tone}\n\n"
                "위 조합으로 현실적인 한국어 스미싱 문자와 라벨을 생성하세요.")
        return [{"role": "system", "content": AUG_SMISHING_SYSTEM},
                {"role": "user", "content": user}]
    category, tone = combo
    user = (f"문자 유형: {category}\n말투: {tone}\n\n"
            "위 유형의 현실적인 '정상' 한국어 문자와 라벨을 생성하세요.")
    return [{"role": "system", "content": AUG_NORMAL_SYSTEM},
            {"role": "user", "content": user}]


def gen_one(client, kind: str, combo, max_retries: int = 3) -> dict | None:
    messages = _messages(kind, combo)
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
            if kind == "smishing":
                rf = [t for t in p.risk_factors if t in RISK_FACTORS]
                return {"message": msg, "verdict": "smishing", "risk_factors": rf,
                        "explanation": p.explanation}
            return {"message": msg, "verdict": "normal", "risk_factors": [],
                    "explanation": p.explanation}
        except Exception:  # noqa: BLE001
            if attempt == max_retries - 1:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def run(kind: str, n: int, limit: int | None, workers: int = 8, seed: int = 42) -> None:
    client = get_openai_client()
    rng = random.Random(seed if kind == "smishing" else seed + 1)  # 종류별 다른 시드
    total = limit or n
    combos = [_combo(kind, rng) for _ in range(total)]

    results: list[dict | None] = [None] * len(combos)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(gen_one, client, kind, c): i for i, c in enumerate(combos)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"증강:{kind}"):
            results[futs[fut]] = fut.result()

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
    fname = "sft_aug.jsonl" if kind == "smishing" else "sft_aug_normal.jsonl"
    out = PROCESSED_DIR / fname
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[{kind}] 생성 {len(recs)}건 저장 (중복 {dropped} 제거) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="합성 스미싱/정상 증강")
    ap.add_argument("--kind", choices=["smishing", "normal"], default="smishing")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--limit", type=int, default=None, help="검증용 상한")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run(args.kind, args.n, args.limit, args.workers)


if __name__ == "__main__":
    main()
