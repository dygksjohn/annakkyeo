# GPU PC 작업 handoff — 공통 세팅

> 3대 GPU PC(RTX 3050 6GB, Windows, CUDA 12.6)에서 병렬 작업을 실행하기 위한 공통 준비.
> **각 PC에서 먼저 이 문서대로 세팅**한 뒤, 해당 PC의 역할 문서를 따른다.
>
> | 머신 | 역할 | 문서 |
> |------|------|------|
> | **개발 PC** | GPU 불필요 작업(데이터 구축·증강·API 베이스라인·EDA). GPU PC의 데이터 공급원 | [개발PC_작업.md](개발PC_작업.md) |
> | PC-1 | 주력 학습 (Qwen2.5-1.5B QLoRA) | [PC1_주력학습.md](PC1_주력학습.md) |
> | PC-2 | 비교 학습 (Qwen2.5-3B / Llama-3.2) | [PC2_비교학습.md](PC2_비교학습.md) |
> | PC-3 | 베이스라인·평가 (LLM-judge 포함) | [PC3_평가.md](PC3_평가.md) |

## 전체 실행 계획 (순서·의존관계)

> **★ 마일스톤 달성 (2026-07-07)**: EXP-004 Qwen3-1.7B QLoRA v1 = 엄격 **F1 0.974** →
> GPT-4o-mini 베이스라인(0.881) **초과**. "1~3B 파인튜닝이 GPT급" 성공 지표 충족.
> 모드 전환: **되게 만들기 → 확정·차별화·포트폴리오 패키징**.

**완료 (Phase 0~3)**: 데이터 구축·증강·EDA·베이스라인 → GPU 세팅 → 병렬 학습 →
주력(Qwen3-1.7B) 평가까지 완료. base zero-shot·QLoRA 비교표는 [평가_실험로그.md](../평가_실험로그.md).

**이후 로드맵**

```
[Phase A] 승리 확정 & 차별점 측정  ← 지금 최우선
  A1 README·리포트 헤드라인 반영 (F1 0.974>GPT)      [개발 PC]  ← 착수
  A2 설명 품질 LLM-judge 측정 (judge_explanation.py)  [PC-3]    ← 핵심 차별점, 미측정
  A3 Kanana QLoRA v1 평가 (EXP-008) → 비교표 완성      [PC-3]
        │
        ▼
[Phase B] v2 개선 (값싸므로 권장)
  B1 v2 재학습 (균형 데이터 950:950, FP·격식체정상 약점) [PC-1]
  B2 v2 평가 → v1(0.974) vs v2 정밀도 비교              [PC-3]
        │
        ▼
[Phase C] 포트폴리오 패키징 (7/14 목표)
  C1  Gradio 데모 (반나절)                             [개발 PC]
  C2  평가 리포트 + HF 모델 카드                        [개발 PC]
  C3a 실 held-out 셋 수집(뉴스·보안공지) + HF 업로드     [개발 PC]  ← GPU 불필요
  C3b held-out 추론·평가 v1/v2 (다른 분포)             [GPU PC: PC-1/PC-3]  ← GPU 필요
```

**핵심 의존관계 / 유의**
- A2·A3는 데이터·어댑터가 이미 있어 **지금 착수 가능** (PC-3)
- B1(v2 재학습)은 개발 PC v2 균형 데이터 업로드 완료 → `hf download` 후 바로 가능
- **C3 분리**: 수집(C3a, 개발 PC) → HF 업로드 → 평가(C3b, GPU PC). held-out 추론은 GPU 필요이므로 개발 PC가 아닌 **어댑터·하네스 보유 PC(PC-1 또는 PC-3)**가 담당. `eval_hf_model.py`에 `--eval-file` 인자 추가 필요(현재 eval_sample 고정)
- ⚠️ 현재 test셋(eval_sample)은 **Kor-Smishing 동일 분포**. F1 0.974는 유효하나, C3(실 held-out)으로 **다른 분포**에서도 GPT급을 보이면 주장이 훨씬 강해짐 → 리포트 전 확보 권장
>
> 에이전트에게: **"이 PC에서는 PC-1 작업을 한다"** 처럼 역할을 지정받으면, 이 공통 세팅을
> 마친 뒤 해당 역할 문서의 절차를 순서대로 실행하라. GPU가 필요한 명령은 실제 하드웨어에서
> 스모크 테스트부터 하고, OOM/버전 오류는 각 문서의 대응 절차로 해결한다.

---

## 0. 전제
- 프로젝트 repo 접근(git), Hugging Face 계정 토큰, (PC-3은) OpenAI 키
- 데이터 원본/파생물은 라이선스상 git 커밋 금지 → **HF private 또는 공유 폴더로 전달**(3절)

## 1. repo 준비
```cmd
git clone <repo-url> annakkyeo
cd annakkyeo
git pull
```

## 2. 파이썬 환경 (Windows CMD)
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft trl bitsandbytes accelerate datasets sentencepiece wandb
pip install -r requirements.txt
```
GPU 인식 확인:
```cmd
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
→ `True` 와 `NVIDIA GeForce RTX 3050` 이 나오면 성공.

> ⚠️ **bitsandbytes Windows 4bit 리스크**: 4bit 로드가 실패하면(대표 증상: `CUDA Setup failed`
> / DLL 오류) 최신화 `pip install -U bitsandbytes` 후 재시도, 그래도 안 되면 **WSL2(Ubuntu)로
> 전환**이 확실한 폴백. 이 판정은 아래 스모크 테스트에서 바로 드러난다.

## 3. `.env` 설정
```cmd
copy .env.example .env
```
`.env` 에 `HF_TOKEN`, `HF_HOME_NAMESPACE`(HF 사용자명), (PC-3만) `OPENAI_API_KEY`,
그리고 wandb 통합을 위해 `WANDB_API_KEY` / `WANDB_ENTITY` / `WANDB_PROJECT=annakkyeo` 기입.

## 4. 학습 데이터 수령
데이터는 개발 PC에서 구축됨(`data/processed/sft_train.jsonl`, `sft_val.jsonl`, `eval_sample.csv`).
Kor-Smishing 파생물이라 git 커밋 불가 → 아래 중 하나로 가져온다.

- **HF private dataset (권장)**: 개발 PC가 `dygksjohn/annakkyeo-sft`(private) 로 업로드해 두면
  ```cmd
  hf download dygksjohn/annakkyeo-sft --repo-type dataset --local-dir data/processed
  ```
- **공유 폴더/USB**: 같은 네트워크이므로 `data/processed/` 폴더째 복사해도 됨.

수령 확인:
```cmd
python -c "import pathlib,json; p=pathlib.Path('data/processed'); print([f.name for f in p.glob('*.jsonl')]); print(sum(1 for _ in open('data/processed/sft_train.jsonl',encoding='utf-8')),'train rows')"
```

## 5. 실험 추적 (선택 — `.env` 의 `REPORT_TO`)
wandb 무료 한도 이슈로 **기본은 TensorBoard(무료·로컬)** 권장.
```cmd
pip install tensorboard
:: 학습 중/후 loss 곡선 보기
tensorboard --logdir outputs
```
- `REPORT_TO=tensorboard` (권장) / `none`(콘솔만) / `wandb`(대시보드, 한도 제한)
- PC 간 최종 비교는 wandb 없이도 **PC-3 가 평가_실험로그.md 에 F1 을 모아** 수행하므로 문제없음.

---
세팅이 끝나면 **자신의 역할 문서**(PC1/PC2/PC3)로 이동한다.
