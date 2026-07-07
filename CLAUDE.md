# CLAUDE.md — annakkyeo (안낚여)

한국어 스미싱 **판정 + 근거 설명**을 출력하는 소형 LLM(1~3B) 파인튜닝 프로젝트.
판정만 하는 스팸 필터와 달리 "왜 위험한지"까지 생성한다. 개인 포트폴리오.

> 이 파일은 모든 PC의 세션에 자동 로딩된다. **세션 시작 시 먼저 `git pull` 하고, 자신의 역할 문서를 읽어라.**

## 이 세션이 도는 머신 확인 (중요)

여러 PC가 같은 repo를 공유한다. 사용자에게 역할을 지정받지 못했으면 **어느 머신인지 먼저 물어라.**

| 머신 | 역할 | 역할 문서 |
|------|------|-----------|
| **개발 PC** | GPU 불필요 작업 — 데이터 구축·증강·API 베이스라인·EDA·문서. GPU PC의 데이터 공급원 | [docs/handoff/개발PC_작업.md](docs/handoff/개발PC_작업.md) |
| **PC-1** | 주력 학습 Qwen3-1.7B QLoRA | [docs/handoff/PC1_주력학습.md](docs/handoff/PC1_주력학습.md) |
| **PC-2** | 비교 학습 Kanana-2.1B / Qwen3-4B | [docs/handoff/PC2_비교학습.md](docs/handoff/PC2_비교학습.md) |
| **PC-3** | 베이스라인·평가(LLM-judge) | [docs/handoff/PC3_평가.md](docs/handoff/PC3_평가.md) |

GPU PC(1/2/3)는 모두 RTX 3050 **6GB**, Windows, CUDA 12.6. 공통 세팅: [docs/handoff/README.md](docs/handoff/README.md).

## 협업 프로토콜 (멀티 PC)

단일 진실 소스 = **이 GitHub repo**. 대용량은 HF Hub(private). 실험 추적은 `.env` 의 `REPORT_TO`(기본 tensorboard, 무료·로컬 / wandb는 한도 제한). PC 간 최종 비교는 PC-3가 평가_실험로그.md 에 F1을 모아 수행.

- 세션 시작: `git pull` → 역할 문서 + [docs/사전준비_체크리스트.md](docs/사전준비_체크리스트.md) 진행 로그로 전체 상황 파악
- 작업 후: **작게 커밋하고 자주 push** (다른 PC가 오래된 상태로 일하지 않게)
- **각 PC는 자기 파일만 수정** (자기 handoff 문서, 자기 outputs). 공용 체크리스트 진행 로그는 **맨 아래에 줄 추가(append-only)** — 머지 충돌 방지
- PC 간 산출물 공유: 데이터·어댑터는 HF private(또는 공유 폴더), git 커밋 금지

## 데이터 취급 원칙 (반드시 준수)

- **원본 데이터·비밀정보 커밋 금지**. `.env`(키), `data/raw/`·`data/processed/`·`outputs/`는 gitignore. 커밋물엔 **집계 지표·통계만**, 문자 원문 금지
- **Kor-Smishing**: 라이선스 미명시 → 파생 데이터셋 공개 배포 전 원저자 문의. 원본 비커밋
- **KISA(CTAS)**: 출처표시+변경금지 → 파생물 공개 배포 금지, 내부 학습·평가만
- **test 누수 금지**: test = `eval_sample.csv`(실데이터). 학습에서 제외됨. test는 실데이터만
- 개인 수신 문자는 발신번호 등 개인정보 제거 후 사용
- 라이선스 상세: [NOTICE](NOTICE)

## 코드 구조 (`src/`)

| 모듈 | 역할 |
|------|------|
| `schema.py` | 출력 스키마·태그(15개)·모델출력 파싱 |
| `prompts.py` | 공용 프롬프트(학습·평가·데모 동일) + 라벨/증강 프롬프트 |
| `metrics.py` | 이진 분류 지표(F1 등) |
| `data.py` | Kor-Smishing 로딩, 평가샘플 |
| `build_sft_data.py` | SFT 학습 데이터 구축(판정+설명) |
| `augment.py` | 합성 스미싱 증강(GPT) |
| `baseline_gpt.py` | GPT API 베이스라인 |
| `train_qlora.py` | QLoRA 학습(PC-1/2) |
| `eval_hf_model.py` | 로컬 모델 평가(PC-3) |
| `eda.py` | Kor-Smishing EDA |

실행은 `python -m src.<module>`. 설계 근거: [docs/스키마_프롬프트_설계.md](docs/스키마_프롬프트_설계.md).

## 핵심 기준선·확정 사항

- **성공 지표**: 파인튜닝 모델 F1이 **GPT-4o-mini 베이스라인 F1 0.881(엄격)** 에 근접·초과 → [docs/평가_실험로그.md](docs/평가_실험로그.md)
- **베이스 모델**: 주력 `Qwen/Qwen3-1.7B`(Apache), 비교 `kakaocorp/kanana-1.5-2.1b-instruct-2505`(Apache), 참고 `EXAONE-4.0-1.2B`(NC, 배포금지) — 계획서 4.1
- **출력 스키마**: `{"verdict":"smishing|suspicious|normal","risk_factors":[...],"explanation":"..."}`

## 관례

- 사용자와의 대화·문서는 **한국어**. CLI 명령은 별도 요청 없으면 사용자 환경(git bash/Windows)에 맞춰 제시
- 전체 배경·계획: [docs/프로젝트_계획서_v1.md](docs/프로젝트_계획서_v1.md). 진행상황 단일 기준: [docs/사전준비_체크리스트.md](docs/사전준비_체크리스트.md)
