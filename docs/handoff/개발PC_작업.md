# 개발 PC — 작업 목록·계획

> **이 노트북(개발 PC)은 GPU 불필요 작업을 전담**하고, GPU PC(PC-1/2/3)에 학습 데이터를 공급한다.
> 도구: OpenAI API(베이스라인·증강·설명라벨·judge), CPU 라이브러리(pandas/sklearn/matplotlib).
> 전제: [README.md](README.md)의 repo·`.env`(OPENAI_API_KEY, HF_TOKEN) 준비.

---

## 작업 목록 (상태)

| # | 작업 | 산출물 | 상태 |
|---|------|--------|------|
| D1 | 스키마·프롬프트·라벨 기준 설계 | [스키마_프롬프트_설계.md](../스키마_프롬프트_설계.md) | ✅ |
| D2 | Kor-Smishing 확보·검증 | `data/raw/` (gitignore) | ✅ |
| D3 | SFT 학습 데이터 구축 (판정+설명) | `data/processed/sft_{train,val}.jsonl` | ✅ 300/80 |
| D4 | GPT API 베이스라인 (EXP-001) | [평가_실험로그.md](../평가_실험로그.md) | ✅ F1 0.881 |
| D5 | Kor-Smishing EDA | [EDA_Kor-Smishing.md](../EDA_Kor-Smishing.md) | ✅ |
| D6 | **증강 파이프라인** (합성 스미싱 생성) | `data/processed/sft_aug.jsonl` | ✅ 800건 (app_install 28%·callback 30%로 EDA 공백 보강). train_qlora가 자동 포함. 개선여지: suspicious_domain·fear_appeal 태그 0% |
| D7 | 전처리 정제 (중복 141건 제거, ham 잡음 필터) | 정제 스크립트 | ⏳ |
| D8 | **SFT+증강 데이터 HF private 업로드** | `{ns}/annakkyeo-sft` | ⏳ (GPU PC 선행조건) |
| D9 | 베이스 모델 최신 확인 (C3) | 계획서 4.1 갱신 | ⏳ |
| D10 | 설명 품질 LLM-judge 스크립트 | `src/judge_explanation.py` | ⏳ (PC-3 평가 지원) |
| D11 | 실데이터 시드 수집 (B2) | `data/seed/*.csv` | ⏳ 상시 |

## 실행 순서 (개발 PC 내부)
```
D6 증강 ──┐
D7 정제 ──┴─→ D8 HF 업로드 ──→ (GPU PC 학습 가능)
D9 모델확인 (독립, 언제든)
D10 judge (PC-3 평가 시작 후)
```

## 명령어 요약
```bash
python -m src.build_sft_data --train 300 --val 80   # D3 (완료)
python -m src.baseline_gpt --n-per-class 300         # D4 (완료)
python -m src.eda                                    # D5 (완료)
python -m src.augment --n 800                        # D6 증강 (합성 스미싱)
# D8 업로드: huggingface-cli upload {ns}/annakkyeo-sft data/processed --repo-type dataset --private
```

## Exit 조건 (GPU 작업 넘기기 전)
- [ ] 증강 포함 학습셋 완성 (스미싱 클래스 수천 건 규모)
- [ ] 전처리로 중복·잡음 제거, train/test 누수 차단 확인
- [ ] 데이터가 HF private(또는 공유폴더)로 PC-1/2/3에서 접근 가능
- [ ] 베이스 모델 후보 최종 확정(C3)
