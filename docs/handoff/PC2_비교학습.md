# PC-2 — 비교 학습 (Qwen2.5-3B / Llama-3.2)

> **이 PC에서는 PC-2 작업을 수행한다.** 목표: 주력(PC-1, 1.5B)과 비교할 모델을
> 병렬로 QLoRA 학습해 "크기·계열에 따른 성능 차이" 표를 만든다.
>
> 전제: [README.md](README.md) 공통 세팅 완료. PC-1 과 **같은 데이터**로 학습(공정 비교).

---

## 우선순위: Qwen2.5-3B (스트레치) → 실패 시 Llama-3.2-1B

### A안. Qwen2.5-3B QLoRA (6GB 빠듯 — 먼저 스모크)
```cmd
python -m src.train_qlora --model Qwen/Qwen2.5-3B-Instruct --max-steps 10 --max-seq 768 --run-name smoke-qwen3b
```
- 통과하면 본 학습:
```cmd
python -m src.train_qlora --model Qwen/Qwen2.5-3B-Instruct --run-name qwen3b-v1 --max-seq 768 --grad-accum 16
```
- **OOM 이면**: `--max-seq 512` → `--lora-r 8` 순. 그래도 안 되면 B안으로.

### B안. Llama-3.2-1B QLoRA (확실히 6GB 안, 계열 비교)
> gated 모델 → HF 에서 라이선스 동의 필수. 공개 배포 시 이름에 "Llama" 포함 요건(NOTICE 참조).
```cmd
python -m src.train_qlora --model meta-llama/Llama-3.2-1B-Instruct --run-name llama1b-v1
```

## 어댑터 공유 & Exit
```cmd
huggingface-cli upload {namespace}/annakkyeo-<run-name> outputs/<run-name> --repo-type model --private
```
- [ ] A안 또는 B안 중 최소 1개 어댑터 완성
- [ ] PC-3 에서 접근 가능
- [ ] wandb 에 PC-1 과 같은 프로젝트로 로그 기록(비교 가능)

## 비교 실험 메모
- 세 모델(1.5B / 3B / Llama-1B)을 **같은 데이터·같은 test셋**으로 평가해야 의미 있음 → 평가는 전부 PC-3 담당
- EXAONE-3.5-2.4B 는 비상업 라이선스라 공개 대상에서 제외, 여력 있으면 zero-shot 비교군으로만(PC-3)

## 막히면
- 3B OOM 이 반복되면 3B 는 **4bit zero-shot 비교(PC-3)** 로만 돌리고 학습은 Llama-1B 로 대체 — 리포트엔 "6GB 제약으로 3B 학습 불가, 추론 비교만"으로 정직하게 기록
- 기타 OOM/버전 대응은 [PC1_주력학습.md](PC1_주력학습.md) 하단과 동일
