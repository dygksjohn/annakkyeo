# PC-3 — 베이스라인·평가 (LLM-judge)

> **이 PC에서는 PC-3 작업을 수행한다.** 목표: 모든 모델을 **동일 test셋**으로 평가해
> 비교표를 완성한다. GPT 베이스라인(F1 0.881)은 이미 있음 → 로컬 모델들을 채운다.
>
> 전제: [README.md](README.md) 공통 세팅 완료 + `.env` 에 `OPENAI_API_KEY`(LLM-judge용).
> test셋 = `eval_sample.csv` (GPT 베이스라인과 동일 → apples-to-apples).

---

## 1. 베이스 모델 zero-shot 베이스라인 (GPU)
계획 4.3의 베이스라인 ①. 파인튜닝 효과를 분리하려면 **같은 프롬프트**로 베이스 모델 측정.
```cmd
python -m src.eval_hf_model --model Qwen/Qwen2.5-1.5B-Instruct --run-name qwen1.5b-zeroshot
python -m src.eval_hf_model --model Qwen/Qwen2.5-3B-Instruct   --run-name qwen3b-zeroshot --limit 200
```
- 먼저 `--limit 20` 으로 생성이 정상인지 스모크 → 정상이면 전체(600)

## 2. 파인튜닝 어댑터 평가 (PC-1/PC-2 산출물)
어댑터를 HF/공유폴더에서 받아 평가:
```cmd
hf download dygksjohn/annakkyeo-qwen1.5b-v1 --local-dir outputs/qwen1.5b-v1
python -m src.eval_hf_model --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/qwen1.5b-v1 --run-name qwen1.5b-qlora-v1
```

## 3. 결과 기록
각 실행이 `outputs/<run-name>/metrics.json` 을 남긴다. 이 숫자들을
[../평가_실험로그.md](../평가_실험로그.md) 의 "측정 예정" 표에 채우고 EXP 번호를 부여.
비교 대상: **GPT 0.881(엄격) 기준선 대비 각 로컬 모델의 zero-shot → QLoRA 개선폭**.

| EXP | 모델 | 방식 | F1(엄격) | 비고 |
|-----|------|------|----------|------|
| 001 | gpt-4o-mini | API zero-shot | 0.881 | 완료 |
| 002 | Qwen2.5-1.5B | zero-shot | ? | PC-3 |
| 004 | Qwen2.5-1.5B | QLoRA | ? | 목표: 001 근접 |

## 4. 설명 품질 LLM-judge (계획 4.3 ②)
판정(F1) 외에 **설명 품질**을 평가 — 루브릭: 사실성(문자 내용과 일치)·구체성·실행가능성.
- 어댑터가 생성한 `explanation` 을 GPT(`OPENAI_JUDGE_MODEL=gpt-4o`)로 1~5점 채점하는
  judge 스크립트가 아직 없음 → **이 PC에서 `src/judge_explanation.py` 를 신규 작성**한다.
  입력: `outputs/<run>/predictions.jsonl`(어댑터 평가 시 저장하도록 eval 확장), 출력: 평균 점수.
- 우선순위: 판정 비교표(1~3절)를 먼저 완성하고, 여력 있을 때 judge 추가.

## Exit 조건
- [ ] Qwen 1.5B zero-shot vs QLoRA F1 비교 완성
- [ ] (가능 시) 3B/Llama 포함 비교표
- [ ] 평가_실험로그.md 갱신 (커밋 가능 — 숫자만)
- [ ] (스트레치) 설명 품질 judge 점수

## 막히면
- 생성이 느림/타임아웃: `--limit` 로 표본 축소해 먼저 숫자 확보 후 전체
- OOM(추론): 4bit 이므로 드묾. 나면 다른 프로세스 GPU 점유 확인
- 어댑터 로드 실패: 베이스 `--model` 이 학습 때와 동일 ID 인지 확인(불일치 시 로드 안 됨)
