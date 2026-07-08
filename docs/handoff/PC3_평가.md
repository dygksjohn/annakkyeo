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
hf download dygksjohn/annakkyeo-qwen3-1.7b-v1 --local-dir outputs/qwen3-1.7b-v1
python -m src.eval_hf_model --model Qwen/Qwen3-1.7B --adapter outputs/qwen3-1.7b-v1 --run-name qwen3-1.7b-qlora-v1
```
> 어댑터의 base `--model` 은 학습 때와 **동일 ID**(Qwen/Qwen3-1.7B)여야 로드됨.

## 3. 결과 기록
각 실행이 `outputs/<run-name>/metrics.json` 을 남긴다. 이 숫자들을
[../평가_실험로그.md](../평가_실험로그.md) 의 "측정 예정" 표에 채우고 EXP 번호를 부여.
비교 대상: **GPT 0.881(엄격) 기준선 대비 각 로컬 모델의 zero-shot → QLoRA 개선폭**.

| EXP | 모델 | 방식 | F1(엄격) | 비고 |
|-----|------|------|----------|------|
| 001 | gpt-4o-mini | API zero-shot | 0.881 | 기준선 |
| 006 | Qwen3-1.7B | zero-shot | 0.722 | ✅ 주력 base |
| 007 | Kanana-2.1B | zero-shot | 0.545 | ✅ 비교 base |
| 004 | Qwen3-1.7B | QLoRA v1 | **0.974** | ✅ ★ GPT 초과 (주력) |
| 009 | Qwen3-1.7B | QLoRA v2 | 0.974 | ✅ 정밀도↑(FP15→10) |
| 008 | Kanana-2.1B | QLoRA v1 | 0.984 | ✅ 비교 |
| 010 | Kanana-2.1B | QLoRA v2 | **0.993** | ✅ ★ 재현율 우선 최고 |
| 011 | Qwen3-4B | QLoRA v1 | 0.985 | ✅ 스케일업 |
| 012 | Qwen3-4B | QLoRA v2 | **0.993** | ✅ ★ 정밀도 1.0·FP0 |
| 002/005 | Qwen2.5-1.5B/3B | zero-shot | 0.486/0.476 | 구 후보(참고) |

전체 표·해석·한계는 [../평가_실험로그.md](../평가_실험로그.md). **결론: 파인튜닝 6종 전부 GPT(0.881) 초과.**

## 4. 설명 품질 LLM-judge (계획 4.3 ②) — ✅ 완료
판정(F1) 외에 **설명 품질**을 평가 — 루브릭: 사실성·구체성·실행가능성 3축 1~5점(gpt-4o).
- `src/judge_explanation.py` 작성 완료. 입력 `outputs/<run>/predictions.jsonl`(eval 시 자동 저장).
- 실행: `python -m src.judge_explanation --run qwen3-1.7b-qlora-v1`
- **결과(EXP-004 설명)**: 사실성 **4.48** / 구체성 2.98 / 실행가능성 3.68, 종합 **3.71/5** (553/600 채점).
  → "판정 F1 0.974 + 설명 사실성 4.48"로 차별점 정량 입증. 약점=구체성(v2 라벨 개선여지).

## Exit 조건 — ✅ 전부 충족 (2026-07-08)
- [x] Qwen3-1.7B zero-shot(0.722) vs QLoRA(v1 0.974·v2 0.974) F1 비교 완성 (주력)
- [x] Kanana-2.1B(zero 0.545 / v1 0.984 / v2 0.993) + Qwen3-4B(v1 0.985 / v2 0.993) 포함 **6종 매트릭스**
- [x] 평가_실험로그.md 갱신 (EXP-004·006~012, 숫자만 커밋)
- [x] 설명 품질 judge 점수 (사실성 4.48)

---

## 실행 결과 요약 (2026-07-08 세션)

**한 일**: base zero-shot 2종(006/007) + QLoRA 어댑터 6종(004/008/009/010/011/012) 공식 600건 평가,
설명 품질 judge(A2), 최종 비교표 완성. 모두 `../평가_실험로그.md`·진행 로그에 기록.

**핵심 결론**
- 파인튜닝 6종 전부 GPT-4o-mini(0.881) **초과**(F1 0.974~0.993).
- 최고점 2개(운영점 상이): **Kanana-2.1B v2**(F1 0.993, 재현율 0.997/FP3) · **Qwen3-4B v2**(F1 0.993, **정밀도 1.0/FP0 무오탐**).
- 정밀도 개선 두 축: ① v2 하드네거티브 재균형(전 모델 FP↓) ② 스케일업(1.7B→4B, FP15→8). 겹치면(4B v2) FP=0.
- **주력 권고**: 1.7B로도 GPT 초과 → 배포·속도상 Qwen3-1.7B 유지, 4B는 "키우면 무오탐" 근거.
- **한계(정직)**: 위는 학습과 동일 출처 분포 held-out. 별도 실 held-out(24건)은 재현율 0.58~0.67로 하락(신유형 미탐, PC-1 C3b) → 일반화 갭. v3에서 신유형 다양화 권장.

**평가한 어댑터(HF private)**: `annakkyeo-{qwen3-1.7b,kanana2.1b,qwen3-4b}-v{1,2}` (base: Qwen3-1.7B / kanana-1.5-2.1b / Qwen3-4B-Instruct-2507)

---

## 환경·재현 노하우 (이 세션에서 확인 — 다음 PC-3 세션 필독)

- **transformers 버전**: Qwen3 로드에 **≥4.51 필수**(4.48은 `model type qwen3` 미인식), Kanana 로드에 **<5.0 필수**(v5 엄격검증이 head_dim 분리형 Llama 거부). 이 세션 확정본 **4.56.2**. `.venv`(Python 3.12) 사용.
- **Qwen3 thinking**: `eval_hf_model.generate()` 에 `enable_thinking=False` 반영됨(없으면 `<think>`가 256토큰 소진→파싱실패 급증). Llama/Kanana 템플릿은 무시하므로 안전.
- **비공개 어댑터 다운로드**: `.env`의 `HF_TOKEN` 필요. `export $(grep ^HF_TOKEN= .env | xargs); hf download <repo> --local-dir outputs/<name>`.
- **한글 콘솔 깨짐(cp949)**: 실행 시 `PYTHONIOENCODING=utf-8` 지정(출력만 깨질 뿐 metrics.json은 UTF-8 정상).
- **base는 자동 다운로드**되나 4B(~8GB)는 시간이 걸림 → 첫 4B 평가 전 `hf download Qwen/Qwen3-4B-Instruct-2507`로 선캐시 권장.
- **평가 소요**: 1.7~2.1B ≈ 11~14초/건, 4B ≈ 14~16초/건 → 600건 약 2~2.7h. 백그라운드 실행 권장(로그는 `| tail` 말고 **파일 리다이렉트**로 남겨야 진행 확인 가능).
- **어댑터 base 확인**: `outputs/<name>/adapter_config.json`의 `base_model_name_or_path` — `--model`에 이 값을 그대로 넣어야 로드됨.

## 막히면
- 생성이 느림/타임아웃: `--limit` 로 표본 축소해 먼저 숫자 확보 후 전체
- OOM(추론): 4bit 이므로 드묾. 나면 다른 프로세스 GPU 점유 확인(`nvidia-smi`)
- 어댑터 로드 실패: 베이스 `--model` 이 학습 때와 동일 ID 인지 확인(불일치 시 로드 안 됨)
- Qwen3 로드 실패(`model type qwen3`): transformers 업그레이드(위 참조)
- 스모크는 `eval_sample.csv` head 20(스미싱14/정상6)이라 F1이 낙관적 → 판단은 전체 600건 기준
