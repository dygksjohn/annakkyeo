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

```
[Phase 0] 개발 PC : 데이터 구축·증강·EDA·베이스라인 ──→ 데이터 HF 업로드
                                                          │ (데이터 공급)
                                                          ▼
[Phase 1] PC-1/2/3 : 공통 세팅(이 문서) + 스모크 테스트
                                                          ▼
[Phase 2] 병렬 학습 :  PC-1(1.5B QLoRA) ┐   PC-3(베이스 zero/few-shot 측정)
                       PC-2(3B/Llama)   ┘        │
                              │ (어댑터 공유)      │
                              └──────────┬────────┘
                                         ▼
[Phase 3] PC-3 : 어댑터 평가 → 비교표(평가_실험로그.md) + 설명 LLM-judge
                                         ▼
[Phase 4] 개발 PC : 최고 모델 선정·Gradio 데모·리포트/HF 공개
```

**핵심 의존관계**
- GPU 학습(Phase 2)은 **개발 PC의 데이터 업로드(Phase 0)** 이후 시작 가능
- PC-3 어댑터 평가(Phase 3)는 **PC-1/2 어댑터 완성** 이후
- PC-3 베이스 zero-shot 측정은 데이터만 있으면 학습과 **병렬 가능**(어댑터 불필요)
- 현재 상태: Phase 0 진행 중 (SFT 데이터·EDA·베이스라인 완료, 증강·업로드 남음)
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

- **HF private dataset (권장)**: 개발 PC가 `{namespace}/annakkyeo-sft`(private) 로 업로드해 두면
  ```cmd
  huggingface-cli download {namespace}/annakkyeo-sft --repo-type dataset --local-dir data/processed
  ```
- **공유 폴더/USB**: 같은 네트워크이므로 `data/processed/` 폴더째 복사해도 됨.

수령 확인:
```cmd
python -c "import pathlib,json; p=pathlib.Path('data/processed'); print([f.name for f in p.glob('*.jsonl')]); print(sum(1 for _ in open('data/processed/sft_train.jsonl',encoding='utf-8')),'train rows')"
```

## 5. wandb 로그인 (실험 추적 통합)
```cmd
wandb login
```
세 PC가 같은 `WANDB_PROJECT=annakkyeo` 로 로그를 보내 실험을 한 곳에서 비교한다.

---
세팅이 끝나면 **자신의 역할 문서**(PC1/PC2/PC3)로 이동한다.
