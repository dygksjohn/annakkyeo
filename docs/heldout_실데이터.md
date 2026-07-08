# held-out 실데이터 셋 (다른 분포 일반화 검증)

> 목적: 학습·평가에 쓴 Kor-Smishing(2018~2021 수집)과 **다른 출처·다른 시점(2024 위주)**의
> 실제 공개 스미싱으로 모델의 일반화(미학습 스미싱 탐지)를 검증.
> **문자 원문은 커밋하지 않음**(gitignore). 데이터는 HF private `dygksjohn/annakkyeo-sft`.
> 이 문서는 출처·구성·주의만 기록(리포트 인용용).

## 구성

| 항목 | 값 |
|------|-----|
| 파일 | `data/processed/heldout_smishing.csv` (HF: `heldout_smishing.csv`) |
| 스키마 | `content, class(=1), category, date` |
| 건수 | **24건 (전부 스미싱)** |
| 유형 | 택배 8 / 지인·부고사칭 6 / 이벤트 3 / 기관사칭 3 / 건강검진 2 / 카드·국제발신 2 |
| 시점 | 2024년 18건 / 2020~2021년 6건 |

## 수집 원칙
- 공개 보안공지·보도자료·언론에 **'스미싱 예시'로 명시 게시된 문구만** 사용 (지어내지 않음)
- 개인정보 마스킹: 실제 URL→`http://***`, 이름→`XXX/정*희`, 번호→`1599-8***`

## 출처
- [미디어오늘 (2024)](https://www.mediatoday.co.kr/news/articleView.html?idxno=315779)
- [보호나라 스미싱 주의보 (2024-01)](https://www.boho.or.kr/kr/bbs/view.do?bbsId=B0000030&menuNo=205027&nttId=71280)
- [정책브리핑 추석 스미싱 (2020)](https://www.korea.kr/news/policyNewsView.do?newsId=148905629)
- [보안뉴스 해외결제 사칭 (2021)](https://m.boannews.com/html/detail.html?idx=102322)
- [신한카드 사칭 안내 (2020)](https://www.shinhancard.com/pconts/html/helpdesk/dataRoom/MOBFM12048/1199110_1127.html)

## 사용법 (C3b — GPU PC)
```bash
hf download dygksjohn/annakkyeo-sft heldout_smishing.csv --repo-type dataset --local-dir data/processed
python -m src.eval_hf_model --model Qwen/Qwen3-1.7B --adapter outputs/qwen3-1.7b-v1 \
  --eval-file data/processed/heldout_smishing.csv --run-name qwen3-1.7b-v1-heldout
```
- 전부 스미싱이므로 **재현율(recall)**만 유의미(정밀도·FP는 eval_sample로 검증됨).
- 결과 해석: "학습에 없던 2024년 실제 스미싱 24건 중 N건 탐지".

## 주의 / 한계
- **소규모(24건)** — 메인 test가 아닌 **실데이터 스팟체크**. 리포트에 규모 명시.
- #14(설 인사)·#16(추석 인사)은 스미싱의 **도입부 후킹 문구**라 단독으론 정상처럼 보임 → 미탐 시 참고.
- 정상(normal) held-out 미포함 → FP 일반화는 별도(본인 수신 문자 익명화, 체크리스트 B2)로 추후 보완.
