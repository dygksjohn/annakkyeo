# PC-1 — 주력 학습 (Qwen3-1.7B QLoRA)

> **이 PC에서는 PC-1 작업을 수행한다.** 목표: 공개용 주력 모델인
> **Qwen3-1.7B**(Apache-2.0) 를 QLoRA 로 파인튜닝해 판정+설명 어댑터를 만든다.
> 6GB VRAM 에 편안한 조합이라 여기서 파이프라인을 먼저 안정화한다.
>
> 전제: [README.md](README.md) 공통 세팅(환경·데이터·wandb) 완료.

---

## 왜 이 모델인가 (2026-07 C3 확정, 계획서 4.1)
- Apache-2.0 → 공개 배포·상업 이용 자유
- 1.7B → 6GB QLoRA 편안(~3.5GB 예상), JSON 지시준수 안정적(구조화 출력에 유리)
- 이 어댑터가 **GPT 베이스라인 F1 0.881(엄격)** 에 근접·초과하는지가 핵심 성공 지표

## 1. 스모크 테스트 (먼저 반드시)
4bit 로드 + 10스텝 학습이 도는지 확인. bitsandbytes Windows 이슈를 여기서 판정.
```cmd
python -m src.train_qlora --model Qwen/Qwen3-1.7B --max-steps 10 --run-name smoke-qwen3-1.7b
```
- 정상: loss 가 찍히고 `outputs/smoke-qwen3-1.7b/` 에 어댑터 저장
- **OOM 이면**: `--max-seq 768` → `--grad-accum 8` 순으로 낮춤
- **CUDA/DLL 오류**: 공통문서 2절의 bitsandbytes 폴백(WSL2) 적용
- **LoRA target 오류**(모듈명 불일치): `train_qlora.py` 의 `TARGET_MODULES` 를 `"all-linear"` 로 바꿔 재시도

## 2. 본 학습
```cmd
python -m src.train_qlora --model Qwen/Qwen3-1.7B --run-name qwen3-1.7b-v1 --epochs 3
```
- 산출물: `outputs/qwen3-1.7b-v1/` (LoRA 어댑터)
- wandb 에서 train/val loss 하락 확인

## 3. 어댑터 공유 (평가는 PC-3)
```cmd
huggingface-cli upload {namespace}/annakkyeo-qwen3-1.7b-v1 outputs/qwen3-1.7b-v1 --repo-type model --private
```
또는 공유 폴더로 `outputs/qwen3-1.7b-v1/` 전달. PC-3 에 "어댑터 준비됨"을 알린다.

## 4. Exit 조건
- [ ] 스모크 통과(4bit 학습 1스텝 이상)
- [ ] 본 학습 완료, val loss 하락 확인
- [ ] 어댑터가 PC-3 에서 접근 가능(HF/공유폴더)
- [ ] (PC-3 평가 후) F1 이 베이스 zero-shot 보다 개선

## 막히면
- OOM: `--max-seq 768 --grad-accum 8`, 그래도면 `--lora-r 8`
- LoRA target 모듈명 불일치: `TARGET_MODULES = "all-linear"` 로 변경
- trl/transformers 버전 인자 불일치: 설치된 버전의 `SFTConfig` 시그니처에 맞춰 인자명 조정
  (예: `eval_strategy` ↔ `evaluation_strategy`, `max_seq_length` 위치)
- 학습이 너무 느림: epochs 를 2 로, 또는 데이터 늘리기 전까진 정상(데이터 380건 소규모, 증강 후 확대 예정)
