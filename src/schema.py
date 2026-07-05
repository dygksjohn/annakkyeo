"""출력 스키마 정의와 모델 출력 파싱.

설계 근거: docs/스키마_프롬프트_설계.md §1
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

VERDICTS = ("smishing", "suspicious", "normal")

# risk_factors 태그 목록 (5개 축, 15개). 설계 문서 §1과 일치.
RISK_FACTORS = frozenset(
    {
        "url_shortener",
        "suspicious_domain",
        "impersonation_delivery",
        "impersonation_gov",
        "impersonation_financial",
        "impersonation_acquaintance",
        "urgency_pressure",
        "fear_appeal",
        "financial_lure",
        "app_install_request",
        "personal_info_request",
        "callback_induce",
        "abnormal_format",
        "unverifiable_sender",
        "link_only_message",
    }
)


@dataclass
class Prediction:
    """파싱된 모델 출력. parse_ok=False 면 형식 파싱 실패."""

    verdict: str
    risk_factors: list[str] = field(default_factory=list)
    explanation: str = ""
    parse_ok: bool = True
    raw: str = ""

    @property
    def is_smishing(self) -> bool:
        """엄격 매핑: verdict == 'smishing' 만 양성."""
        return self.verdict == "smishing"

    @property
    def is_flagged(self) -> bool:
        """관대 매핑: suspicious 도 양성(경고)으로 취급."""
        return self.verdict in ("smishing", "suspicious")


def _extract_json(text: str) -> dict | None:
    """텍스트에서 첫 JSON 오브젝트를 추출. 실패 시 None."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def parse_model_output(text: str) -> Prediction:
    """모델의 원시 문자열 출력을 Prediction 으로 정규화."""
    obj = _extract_json(text)
    if obj is None or not isinstance(obj, dict):
        return Prediction(verdict="normal", parse_ok=False, raw=text or "")

    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        return Prediction(verdict="normal", parse_ok=False, raw=text or "")

    rf = obj.get("risk_factors", []) or []
    if not isinstance(rf, list):
        rf = []
    rf = [str(x).strip() for x in rf]

    return Prediction(
        verdict=verdict,
        risk_factors=rf,
        explanation=str(obj.get("explanation", "")).strip(),
        parse_ok=True,
        raw=text or "",
    )
