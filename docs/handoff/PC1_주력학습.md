# PC-1 — 주력 학습 (Qwen2.5-1.5B QLoRA)

> **이 PC에서는 PC-1 작업을 수행한다.** 목표: 공개용 주력 모델인
> Qwen2.5-1.5B-Instruct 를 QLoRA 로 파인튜닝해 판정+설명 어댑터를 만든다.
> 6GB VRAM 에 가장 편안한 조합이라 여기서 파이프라인을 먼저 안정화한다.
>
> 전제: [README.md](README.md) 공통 세팅(환경·데이터·wandb) 완료.

---

## 왜 이 모델인가
- Apache-2.0 → 공개 배포 자유(계획서 6장 리스크 대응)
- 한국어 준수, 1.5B → 6GB QLoRA 여유(~3.5GB 사용 예상)
- 이 어댑터가 **GPT 베이스라인 F1 0.881(엄격)** 에 근접·초과하는지가 핵심 성공 지표

## 1. 스모크 테스트 (먼저 반드시)
4bit 로드 + 10스텝 학습이 도는지 확인. bitsandbytes Windows 이슈를 여기서 판정.
```cmd
python -m src.train_qlora --model Qwen/Qwen2.5-1.5B-Instruct --max-steps 10 --run-name smoke-qwen1.5b
```
- 정상: loss 가 찍히고 `outputs/smoke-qwen1.5b/` 에 어댑터 저장
- **OOM 이면**: `--max-seq 768` → `--grad-accum 8` 순으로 낮춤
- **CUDA/DLL 오류**: 공통문서 2절의 bitsandbytes 폴백(WSL2) 적용

## 2. 본 학습
```cmd
python -m src.train_qlora --model Qwen/Qwen2.5-1.5B-Instruct --run-name qwen1.5b-v1 --epochs 3
```
- 산출물: `outputs/qwen1.5b-v1/` (LoRA 어댑터)
- wandb 에서 train/val loss 하락 확인

## 3. 어댑터 공유 (평가는 PC-3)
```cmd
huggingface-cli upload {namespace}/annakkyeo-qwen1.5b-v1 outputs/qwen1.5b-v1 --repo-type model --private
```
또는 공유 폴더로 `outputs/qwen1.5b-v1/` 전달. PC-3 에 "어댑터 준비됨"을 알린다.

## 4. Exit 조건
- [ ] 스모크 통과(4bit 학습 1스텝 이상)
- [ ] 본 학습 완료, val loss 하락 확인
- [ ] 어댑터가 PC-3 에서 접근 가능(HF/공유폴더)
- [ ] (PC-3 평가 후) F1 이 베이스 zero-shot 보다 개선

## 막히면
- OOM: `--max-seq 768 --grad-accum 8`, 그래도면 `--lora-r 8`
- trl/transformers 버전 인자 불일치: 설치된 버전의 `SFTConfig` 시그니처에 맞춰 인자명 조정
  (예: `eval_strategy` ↔ `evaluation_strategy`, `max_seq_length` 위치)
- 학습이 너무 느림: epochs 를 2 로, 또는 데이터 늘리기 전까진 정상(데이터 380건 소규모)
