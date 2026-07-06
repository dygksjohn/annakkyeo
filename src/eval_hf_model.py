"""로컬 HF 모델 평가 (PC-3). 베이스 zero-shot 또는 QLoRA 어댑터를 test셋에 평가.

test셋 = EXP-001 과 동일한 eval_sample.csv (GPT 베이스라인과 apples-to-apples).
평가 지표·파싱은 GPT 베이스라인과 동일 모듈 재사용.

⚠️ GPU 필요. 개발 PC에서 작성 → 실제 GPU에서 스모크 테스트 필요.

베이스 zero-shot:  python -m src.eval_hf_model --model Qwen/Qwen2.5-1.5B-Instruct --run-name qwen1.5b-zeroshot
어댑터 평가:      python -m src.eval_hf_model --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/qwen1.5b-v1 --run-name qwen1.5b-qlora-v1
"""
from __future__ import annotations

import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import OUTPUT_DIR
from .data import get_eval_sample
from .metrics import binary_report, format_report
from .prompts import build_messages
from .schema import parse_model_output


def load_model(model_id: str, adapter: str | None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, text: str, max_new_tokens: int = 256) -> str:
    messages = build_messages(text)
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(
        inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


def run(model_id: str, adapter: str | None, run_name: str, limit: int | None) -> dict:
    tok, model = load_model(model_id, adapter)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    df = get_eval_sample()
    if limit:
        df = df.head(limit)
    y_true = df["class"].astype(int).tolist()

    preds = []
    for text in tqdm(df["content"].tolist(), desc=run_name):
        preds.append(parse_model_output(generate(tok, model, text)))

    y_strict = [1 if p.is_smishing else 0 for p in preds]
    y_lenient = [1 if p.is_flagged else 0 for p in preds]
    parse_fail = sum(1 for p in preds if not p.parse_ok)
    strict = binary_report(y_true, y_strict)
    lenient = binary_report(y_true, y_lenient)

    out_dir = OUTPUT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 원문은 남기지 않고 예측만 저장(설명 품질 judge 입력용). orig_index 로 원본 추적.
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
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
        "model": model_id, "adapter": adapter, "n": len(df),
        "parse_failures": parse_fail,
        "strict_smishing_only": strict, "lenient_incl_suspicious": lenient,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"{run_name}  (adapter={adapter}, parse 실패={parse_fail}/{len(df)})")
    print("=" * 60)
    print(format_report("엄격", strict))
    print("-" * 60)
    print(format_report("관대", lenient))
    print(f"\n저장: {out_dir/'metrics.json'}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="로컬 HF 모델 평가")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="QLoRA 어댑터 경로(없으면 베이스 zero-shot)")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.model, args.adapter, args.run_name, args.limit)


if __name__ == "__main__":
    main()
