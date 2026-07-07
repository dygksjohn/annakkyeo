"""Kor-Smishing 탐색적 데이터 분석 (EDA).

집계 통계와 차트(영문 라벨)만 산출한다. 문자 원문은 저장/게시하지 않는다
(라이선스·개인정보 원칙). 차트는 docs/assets/eda/ 에 저장(커밋 가능).

실행: python -m src.eda
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ROOT
from .data import load_kor_smishing

FIG_DIR = ROOT / "docs" / "assets" / "eda"

URL_RE = re.compile(r"(?:https?://|www\.|\b[\w-]+\.(?:com|net|kr|co|xyz|top|monster|su|link|click|shop|vip|cn|ru)\b)", re.I)
INTL_RE = re.compile(r"국외발신|국제발신|해외발신|\[Web발신\]|\[국외\]")
SHORTENER_RE = re.compile(r"(?:bit\.ly|han\.gl|me2\.kr|tinyurl|goo\.gl|is\.gd|buly\.kr|url\.kr|abr\.ge|c11\.kr)", re.I)

# risk_factor 태그 존재 여부를 근사 확인하는 키워드 (태그 체계 검증용)
TAG_KEYWORDS = {
    "impersonation_delivery": r"택배|배송|물품|송장|운송장|우체국|CJ|한진|롯데|우편",
    "impersonation_gov": r"경찰|검찰|법원|국세청|건강보험|질병관리|정부|과태료|교통민원|이파인|범칙금",
    "impersonation_financial": r"은행|카드|대출|계좌|금융|신용|카카오뱅크|국민|신한|우리|하나",
    "app_install_request": r"설치|앱|APK|어플|다운로드|업데이트",
    "personal_info_request": r"인증|비밀번호|주민등록|계좌번호|본인확인|개인정보",
    "urgency_pressure": r"즉시|지금|오늘|긴급|마감|서둘|빨리|기한",
    "fear_appeal": r"연체|미납|정지|압류|고소|처벌|피해|범죄|경고",
    "financial_lure": r"당첨|지원금|환급|무료|혜택|이벤트|사은품|캐시백|포인트",
}


def pct(n, d):
    return round(100 * n / d, 1) if d else 0.0


def main() -> None:
    df = load_kor_smishing()
    df["len"] = df["content"].str.len()
    df["has_url"] = df["content"].str.contains(URL_RE)
    df["has_intl"] = df["content"].str.contains(INTL_RE)
    df["has_shortener"] = df["content"].str.contains(SHORTENER_RE)

    smish = df[df["class"] == 1]
    ham = df[df["class"] == 0]
    n_s, n_h = len(smish), len(ham)

    print(f"총 {len(df):,}건 | 정상 {n_h:,} / 스미싱 {n_s:,} | 불균형 {n_h/n_s:.1f}:1")
    print(f"중복(정확일치) content: 전체 {df['content'].duplicated().sum():,}건")
    print("\n[길이] (문자수)  중앙값 / 평균 / 95pct")
    for name, s in (("정상", ham), ("스미싱", smish)):
        print(f"  {name}: {s['len'].median():.0f} / {s['len'].mean():.0f} / {s['len'].quantile(0.95):.0f}")
    print(f"  아주 짧은(≤5자) 정상: {pct((ham['len']<=5).sum(), n_h)}%  ← ham 잡음 지표")

    print("\n[URL/발신] 포함율 (%)")
    print(f"{'지표':<14}{'정상':>8}{'스미싱':>8}")
    for label, col in (("URL", "has_url"), ("단축URL", "has_shortener"), ("국외/Web발신", "has_intl")):
        print(f"{label:<14}{pct(ham[col].sum(),n_h):>8}{pct(smish[col].sum(),n_s):>8}")

    print("\n[risk_factor 태그 키워드 출현율] (스미싱 기준, %)")
    tag_rates = {}
    for tag, pat in TAG_KEYWORDS.items():
        r = pct(smish["content"].str.contains(pat, regex=True).sum(), n_s)
        tag_rates[tag] = r
        print(f"  {tag:<26}{r:>6}")

    # ── 차트 ───────────────────────────────────────────────
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 클래스 분포 (로그 스케일)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(["ham (normal)", "smishing"], [n_h, n_s], color=["#4C78A8", "#E45756"])
    ax.set_yscale("log"); ax.set_ylabel("count (log)")
    ax.set_title(f"Class distribution (imbalance {n_h/n_s:.0f}:1)")
    for i, v in enumerate([n_h, n_s]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(FIG_DIR / "class_distribution.png", dpi=120); plt.close(fig)

    # 2) 길이 분포 (히스토그램, 0~300자)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(ham["len"].clip(upper=300), bins=40, alpha=0.6, label="ham", color="#4C78A8", density=True)
    ax.hist(smish["len"].clip(upper=300), bins=40, alpha=0.6, label="smishing", color="#E45756", density=True)
    ax.set_xlabel("message length (chars)"); ax.set_ylabel("density")
    ax.set_title("Message length by class"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "length_by_class.png", dpi=120); plt.close(fig)

    # 3) URL/발신 포함율
    fig, ax = plt.subplots(figsize=(5, 3.2))
    labels = ["URL", "shortener", "intl/Web"]
    hv = [pct(ham[c].sum(), n_h) for c in ("has_url", "has_shortener", "has_intl")]
    sv = [pct(smish[c].sum(), n_s) for c in ("has_url", "has_shortener", "has_intl")]
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], hv, width=0.4, label="ham", color="#4C78A8")
    ax.bar([i + 0.2 for i in x], sv, width=0.4, label="smishing", color="#E45756")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_ylabel("% of messages")
    ax.set_title("URL / sender markers by class"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "url_markers_by_class.png", dpi=120); plt.close(fig)

    print(f"\n차트 저장 → {FIG_DIR}")


if __name__ == "__main__":
    main()
