"""QLoRA SFT 학습 (PC-1/PC-2 공용). --model 로 대상 모델을 바꾼다.

RTX 3050 6GB 최적화 기본값: 4bit(nf4) + batch1 + grad accum + gradient checkpointing
+ paged_adamw_8bit. 문자가 짧아 max_seq_length 1024 로 충분.

⚠️ 이 스크립트는 GPU가 없는 개발 PC에서 작성되어 **실제 GPU에서 스모크 테스트 필요**.
   trl/transformers 버전에 따라 인자명이 다를 수 있음(설치 버전에 맞춰 조정).

스모크:  python -m src.train_qlora --model Qwen/Qwen2.5-1.5B-Instruct --max-steps 10
본학습:  python -m src.train_qlora --model Qwen/Qwen2.5-1.5B-Instruct --run-name qwen1.5b-v1
"""
from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from .config import OUTPUT_DIR, PROCESSED_DIR

# Qwen2.5 / Llama-3.2 공통 LoRA 대상 모듈
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA SFT 학습")
    ap.add_argument("--model", required=True, help="HF 모델 ID (예: Qwen/Qwen2.5-1.5B-Instruct)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=-1, help="스모크용 상한(-1=무제한)")
    args = ap.parse_args()

    run_name = args.run_name or args.model.split("/")[-1].lower() + "-qlora"
    out_dir = OUTPUT_DIR / run_name
    use_wandb = bool(os.getenv("WANDB_API_KEY")) and os.getenv("WANDB_MODE") != "disabled"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
    )

    # train = 실데이터(sft_train) + 증강(sft_aug, 있으면). val = 실데이터만.
    train_files = [str(PROCESSED_DIR / "sft_train.jsonl")]
    aug = PROCESSED_DIR / "sft_aug.jsonl"
    if aug.exists():
        train_files.append(str(aug))
        print(f"증강 데이터 포함: {aug.name}")
    ds = load_dataset(
        "json",
        data_files={"train": train_files, "validation": str(PROCESSED_DIR / "sft_val.jsonl")},
    )

    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        max_seq_length=args.max_seq,
        packing=False,
        report_to="wandb" if use_wandb else "none",
        run_name=run_name,
    )

    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        peft_config=peft_config, processing_class=tok,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"\n✅ 어댑터 저장: {out_dir}")
    print("다음: PC-3 에서 평가  →  python -m src.eval_hf_model "
          f"--model {args.model} --adapter {out_dir}")


if __name__ == "__main__":
    main()
