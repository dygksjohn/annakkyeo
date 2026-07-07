"""설명 품질 LLM-judge (PC-3, 계획 4.3 ②).

판정(F1) 외에 모델이 생성한 `explanation` 의 **품질**을 GPT judge 로 1~5점 채점한다.
루브릭 3축:
  - 사실성(factuality)  : 설명이 실제 문자 내용과 일치하는가(환각/과장 없이)
  - 구체성(specificity) : 문자의 어떤 부분이 왜 위험/안전한지 구체적으로 짚는가
  - 실행가능성(actionability): 사용자가 앞으로 스스로 판별할 일반 원칙을 주는가

입력 : outputs/<run>/predictions.jsonl  (eval_hf_model / baseline_gpt 산출물)
        predictions.jsonl 에는 원문이 없으므로 orig_index 로 eval_sample.csv 를 로컬 조인해
        문자 내용을 가져온다(원문은 judge 입력으로만 쓰고 저장하지 않음).
출력 : outputs/<run>/judge.json  (축별 평균·전체 평균·채점 건수 — 숫자만, 커밋 가능)

실행:
    python -m src.judge_explanation --run qwen1.5b-qlora-v1
    python -m src.judge_explanation --run qwen1.5b-qlora-v1 --limit 50   # 표본만
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .config import JUDGE_MODEL, OUTPUT_DIR, get_openai_client
from .data import get_eval_sample

AXES = ("factuality", "specificity", "actionability")

JUDGE_SYSTEM = (
    "당신은 한국어 스미싱 판정 설명의 품질을 평가하는 엄격한 채점자입니다.\n"
    "문자 원문, 모델의 판정(verdict), 모델의 설명(explanation)이 주어집니다.\n"
    "설명을 아래 3축으로 각각 1~5 정수로 채점하세요.\n"
    "- factuality(사실성): 설명이 실제 문자 내용과 일치하는가. 문자에 없는 내용을 지어내면 감점.\n"
    "- specificity(구체성): 문자의 어떤 부분이 왜 위험/안전한지 구체적으로 짚는가.\n"
    "- actionability(실행가능성): 사용자가 앞으로 스스로 판별할 수 있는 일반 원칙을 담는가.\n"
    "반드시 아래 JSON 형식으로만 답하세요.\n"
    '{"factuality": <1-5>, "specificity": <1-5>, "actionability": <1-5>}'
)


def _build_judge_messages(content: str, verdict: str, explanation: str) -> list[dict]:
    user = (
        f"문자 원문: {content}\n\n"
        f"모델 판정(verdict): {verdict}\n"
        f"모델 설명(explanation): {explanation}\n\n"
        "위 설명을 3축으로 채점해 JSON 으로만 답하세요."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _score_one(client, model: str, content: str, verdict: str, explanation: str,
               max_retries: int = 3) -> dict | None:
    """단일 설명 채점. 실패 시 None."""
    messages = _build_judge_messages(content, verdict, explanation)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0, seed=42,
                response_format={"type": "json_object"},
            )
            obj = json.loads(resp.choices[0].message.content or "{}")
            out = {}
            for ax in AXES:
                v = int(obj.get(ax, 0))
                if not 1 <= v <= 5:
                    return None
                out[ax] = v
            return out
        except Exception:  # noqa: BLE001 - API/파싱 오류 재시도
            if attempt == max_retries - 1:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def run(run_name: str, limit: int | None = None, workers: int = 6,
        model: str = JUDGE_MODEL) -> dict:
    run_dir = OUTPUT_DIR / run_name
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} 없음. 먼저 eval 을 실행해 predictions.jsonl 을 생성하세요."
        )

    preds = [json.loads(line) for line in pred_path.open(encoding="utf-8")]

    # orig_index → 문자 원문 (로컬 조인, 저장하지 않음)
    df = get_eval_sample()
    content_by_idx = {
        int(r["orig_index"]): str(r["content"]) for _, r in df.iterrows()
    }

    # 채점 대상: 형식 파싱 성공 + 비어있지 않은 설명만
    targets = [
        p for p in preds
        if p.get("parse_ok") and str(p.get("explanation", "")).strip()
    ]
    n_total = len(preds)
    n_no_expl = n_total - len(targets)
    if limit:
        targets = targets[:limit]

    client = get_openai_client()
    scores: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                _score_one, client, model,
                content_by_idx.get(int(p["orig_index"]), ""),
                p.get("verdict", ""), p["explanation"],
            ): p
            for p in targets
            if content_by_idx.get(int(p["orig_index"]))
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"judge({model})"):
            s = fut.result()
            if s is not None:
                scores.append(s)

    n_judged = len(scores)
    means = {
        ax: round(sum(s[ax] for s in scores) / n_judged, 3) if n_judged else None
        for ax in AXES
    }
    overall = (
        round(sum(means[ax] for ax in AXES) / len(AXES), 3)
        if n_judged else None
    )
    result = {
        "run": run_name,
        "judge_model": model,
        "n_predictions": n_total,
        "n_no_explanation": n_no_expl,   # 파싱실패/빈 설명 → 채점 제외
        "n_judged": n_judged,
        "axis_means": means,
        "overall_mean": overall,
    }
    with open(run_dir / "judge.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"설명 품질 judge  run={run_name}  judge={model}")
    print("=" * 60)
    print(f"채점 {n_judged} / 예측 {n_total}건 (설명없음/파싱실패 {n_no_expl} 제외)")
    for ax in AXES:
        print(f"  {ax:14s}: {means[ax]}")
    print(f"  {'overall':14s}: {overall}")
    print(f"\n저장: {run_dir/'judge.json'}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="설명 품질 LLM-judge")
    ap.add_argument("--run", required=True, help="outputs/ 아래 run 이름")
    ap.add_argument("--limit", type=int, default=None, help="채점 표본 수(설명 있는 것 기준)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default=JUDGE_MODEL)
    args = ap.parse_args()
    run(args.run, limit=args.limit, workers=args.workers, model=args.model)


if __name__ == "__main__":
    main()
