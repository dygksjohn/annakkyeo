# PC-3 — 베이스라인·평가 (LLM-judge)

> **이 PC에서는 PC-3 작업을 수행한다.** 목표: 모든 모델을 **동일 test셋**으로 평가해
> 비교표를 완성한다. GPT 베이스라인(F1 0.881)은 이미 있음 → 로컬 모델들을 채운다.
>
> 전제: [README.md](README.md) 공통 세팅 완료 + `.env` 에 `OPENAI_API_KEY`(LLM-judge용).
> test셋 = `eval_sample.csv` (GPT 베이스라인과 동일 → apples-to-apples).

---

## 1. 베이스 모델 zero-shot 베이스라인 (GPU)
계획 4.3의 베이스라인 ①. 파인튜닝 효과를 분리하려면 **같은 프롬프트**로 베이스 모델 측정.
**측정 대상 = C3 재확정 베이스 모델**(주력 Qwen3-1.7B / 비교 Kanana-2.1B) — PC-1/PC-2가 실제 학습하는 모델과 일치시켜야 apples-to-apples.
```cmd
python -m src.eval_hf_model --model Qwen/Qwen3-1.7B --run-name qwen3-1.7b-zeroshot
python -m src.eval_hf_model --model kakaocorp/kanana-1.5-2.1b-instruct-2505 --run-name kanana-2.1b-zeroshot
```
- 먼저 `--limit 20` 으로 생성이 정상인지 스모크 → 정상이면 전체(600)
- ⚠️ **transformers ≥ 4.51 필수**(Qwen3 아키텍처 지원). 4.48.3에서는 `model type qwen3 ... not recognize` 로 로드 실패 → `pip install -U "transformers>=4.51"`. **PC-1/PC-2도 Qwen3 학습에 동일 요건**.
- Qwen3는 하이브리드 추론 모델 → eval 은 `enable_thinking=False`(코드 반영)로 `<think>` 블록을 꺼야 JSON 정상 출력(안 끄면 파싱 실패 급증).
- 구 후보(Qwen2.5-1.5B/3B) zero-shot 은 EXP-002/005 에 참고로 남김(모델 확정 변경 전 측정).

## 2. 파인튜닝 어댑터 평가 (PC-1/PC-2 산출물)
어댑터를 HF/공유폴더에서 받아 평가:
```cmd
huggingface-cli download {namespace}/annakkyeo-qwen3-1.7b-v1 --local-dir outputs/qwen3-1.7b-v1
python -m src.eval_hf_model --model Qwen/Qwen3-1.7B --adapter outputs/qwen3-1.7b-v1 --run-name qwen3-1.7b-qlora-v1
```
> 어댑터의 base `--model` 은 학습 때와 **동일 ID**(Qwen/Qwen3-1.7B)여야 로드됨.

## 3. 결과 기록
각 실행이 `outputs/<run-name>/metrics.json` 을 남긴다. 이 숫자들을
[../평가_실험로그.md](../평가_실험로그.md) 의 "측정 예정" 표에 채우고 EXP 번호를 부여.
비교 대상: **GPT 0.881(엄격) 기준선 대비 각 로컬 모델의 zero-shot → QLoRA 개선폭**.

| EXP | 모델 | 방식 | F1(엄격) | 비고 |
|-----|------|------|----------|------|
| 001 | gpt-4o-mini | API zero-shot | 0.881 | 완료 (기준선) |
| 006 | Qwen3-1.7B | zero-shot | ? | PC-3, 주력 base |
| 007 | Kanana-2.1B | zero-shot | ? | PC-3, 비교 base |
| 004 | Qwen3-1.7B | QLoRA | ? | 목표: 001 근접 (006 대비 개선폭) |
| 002/005 | Qwen2.5-1.5B/3B | zero-shot | 0.486/0.476 | 구 후보(참고) |

## 4. 설명 품질 LLM-judge (계획 4.3 ②)
판정(F1) 외에 **설명 품질**을 평가 — 루브릭: 사실성(문자 내용과 일치)·구체성·실행가능성.
- 어댑터가 생성한 `explanation` 을 GPT(`OPENAI_JUDGE_MODEL=gpt-4o`)로 1~5점 채점하는
  judge 스크립트가 아직 없음 → **이 PC에서 `src/judge_explanation.py` 를 신규 작성**한다.
  입력: `outputs/<run>/predictions.jsonl`(어댑터 평가 시 저장하도록 eval 확장), 출력: 평균 점수.
- 우선순위: 판정 비교표(1~3절)를 먼저 완성하고, 여력 있을 때 judge 추가.

## Exit 조건
- [ ] Qwen3-1.7B zero-shot vs QLoRA F1 비교 완성 (주력)
- [ ] (가능 시) Kanana-2.1B zero-shot/QLoRA 포함 비교표
- [ ] 평가_실험로그.md 갱신 (커밋 가능 — 숫자만)
- [ ] (스트레치) 설명 품질 judge 점수

## 막히면
- 생성이 느림/타임아웃: `--limit` 로 표본 축소해 먼저 숫자 확보 후 전체
- OOM(추론): 4bit 이므로 드묾. 나면 다른 프로세스 GPU 점유 확인
- 어댑터 로드 실패: 베이스 `--model` 이 학습 때와 동일 ID 인지 확인(불일치 시 로드 안 됨)
