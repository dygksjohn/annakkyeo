# PC-2 — 비교 학습 (Kanana 1.5 / EXAONE 4.0)

> **이 PC에서는 PC-2 작업을 수행한다.** 목표: 주력(PC-1, Qwen3-1.7B)과 비교할
> **한국어 특화 모델**을 병렬 QLoRA 학습해 "계열에 따른 성능 차이" 표를 만든다.
>
> 전제: [README.md](README.md) 공통 세팅 완료. PC-1 과 **같은 데이터**로 학습(공정 비교).
> 모델 확정 근거: 계획서 4.1 (2026-07 C3 재조사).

---

## 우선순위: Kanana 1.5-2.1B (한국어 특화, 공개 가능) → 여력 시 Qwen3-4B

### A안. Kanana 1.5-2.1B QLoRA (주 비교군, Apache-2.0)
드물게 **Apache-2.0 + 한국어 특화**를 만족하는 국산 모델. 스미싱 은어·축약에 강할 여지.
```cmd
python -m src.train_qlora --model kakaocorp/kanana-1.5-2.1b-instruct-2505 --max-steps 10 --run-name smoke-kanana2.1b
python -m src.train_qlora --model kakaocorp/kanana-1.5-2.1b-instruct-2505 --run-name kanana2.1b-v1
```
- LoRA target 오류 시 `TARGET_MODULES="all-linear"` (Kanana 아키텍처 모듈명이 다를 수 있음)

### B안. Qwen3-4B-Instruct-2507 (품질 상향, 6GB 빠듯)
```cmd
python -m src.train_qlora --model Qwen/Qwen3-4B-Instruct-2507 --max-steps 10 --max-seq 768 --run-name smoke-qwen3-4b
```
- 통과 시 본 학습(`--max-seq 768 --grad-accum 16`). OOM 이면 `--max-seq 512 --lora-r 8`, 그래도 안 되면 A안만.

## 성능 상한 참고 (⚠️ 배포 금지, 내부 벤치용만)
`LGAI-EXAONE/EXAONE-4.0-1.2B` 는 **비상업(NC) 라이선스** → 어댑터 공개·상업 배포 불가.
"국산 SOTA 소형이 이 태스크에서 어디까지 되나" 내부 기준선으로만. 학습보다 **PC-3의 zero-shot 비교**로 돌리는 편이 안전.

## 어댑터 공유 & Exit
```cmd
huggingface-cli upload {namespace}/annakkyeo-<run-name> outputs/<run-name> --repo-type model --private
```
- [ ] Kanana(A안) 어댑터 완성 (필수)
- [ ] 여력 시 Qwen3-4B(B안)
- [ ] PC-3 에서 접근 가능 (loss 추적은 `REPORT_TO`, 최종 비교는 PC-3 평가표)

## 비교 실험 원칙
- 모든 모델을 **같은 데이터·같은 test셋(eval_sample)**으로 평가해야 의미 있음 → 평가는 전부 PC-3
- 배포 후보(Apache): Qwen3-1.7B(PC-1), Kanana-2.1B, Qwen3-4B / 참고용(NC): EXAONE-4.0

## 막히면
- OOM 반복 시 해당 모델은 **4bit zero-shot 비교(PC-3)** 로만 돌리고 리포트에 "6GB 제약"으로 정직히 기록
- 기타 OOM/버전/target 대응은 [PC1_주력학습.md](PC1_주력학습.md) 하단과 동일
