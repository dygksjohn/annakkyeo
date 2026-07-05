"""GPT API 제로샷 베이스라인.

계획 4.3의 베이스라인 3종 중 'GPT API' 판정 성능을 측정한다.
여기서 만든 평가 흐름(프롬프트 → 파싱 → 지표)은 파인튜닝 모델 평가에 재사용된다.

실행:
    python -m src.baseline_gpt --n-per-class 100
결과:
    outputs/gpt_baseline/predictions.jsonl  (원문 포함 → 커밋 금지)
    outputs/gpt_baseline/metrics.json
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .config import BASELINE_MODEL, OUTPUT_DIR, get_openai_client
from .data import get_eval_sample
from .metrics import binary_report, format_report
from .prompts import build_messages
from .schema import parse_model_output

OUT_DIR = OUTPUT_DIR / "gpt_baseline"


def classify_one(client, model: str, text: str, max_retries: int = 3) -> str:
    """단일 문자 판정. 원시 문자열(JSON) 반환. 재시도 포함."""
    messages = build_messages(text)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                seed=42,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 - API 오류 재시도
            if attempt == max_retries - 1:
                return json.dumps({"error": str(e)})
            time.sleep(2 * (attempt + 1))
    return ""


def run(n_per_class: int = 100, model: str = BASELINE_MODEL, workers: int = 6) -> dict:
    client = get_openai_client()
    df = get_eval_sample(n_per_class=n_per_class)
    texts = df["content"].tolist()
    y_true = df["class"].astype(int).tolist()

    raws: list[str | None] = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(classify_one, client, model, t): i for i, t in enumerate(texts)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"GPT({model})"):
            raws[futures[fut]] = fut.result()

    preds = [parse_model_output(r or "") for r in raws]
    y_strict = [1 if p.is_smishing else 0 for p in preds]
    y_lenient = [1 if p.is_flagged else 0 for p in preds]
    parse_fail = sum(1 for p in preds if not p.parse_ok)

    strict = binary_report(y_true, y_strict)
    lenient = binary_report(y_true, y_lenient)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "predictions.jsonl", "w", encoding="utf-8") as f:
        for i, p in enumerate(preds):
            f.write(
                json.dumps(
                    {
                        "orig_index": int(df.iloc[i].get("orig_index", i)),
                        "true": y_true[i],
                        "verdict": p.verdict,
                        "risk_factors": p.risk_factors,
                        "explanation": p.explanation,
                        "parse_ok": p.parse_ok,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = {
        "model": model,
        "n_per_class": n_per_class,
        "parse_failures": parse_fail,
        "strict_smishing_only": strict,
        "lenient_incl_suspicious": lenient,
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"GPT 베이스라인 결과  (model={model}, parse 실패={parse_fail}/{len(preds)})")
    print("=" * 60)
    print(format_report("엄격: smishing 만 양성", strict))
    print("-" * 60)
    print(format_report("관대: suspicious 포함 양성", lenient))
    print(f"\n저장: {OUT_DIR/'metrics.json'}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="GPT API 스미싱 판정 베이스라인")
    ap.add_argument("--n-per-class", type=int, default=100, help="클래스별 샘플 수")
    ap.add_argument("--model", type=str, default=BASELINE_MODEL)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    run(n_per_class=args.n_per_class, model=args.model, workers=args.workers)


if __name__ == "__main__":
    main()
