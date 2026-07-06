"""학습·평가·데모 공용 프롬프트 템플릿.

설계 근거: docs/스키마_프롬프트_설계.md §2
train/serve 불일치를 막기 위해 모든 경로에서 이 템플릿을 사용한다.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "당신은 한국어 문자 메시지의 스미싱(문자 사기) 여부를 분석하는 보안 도우미입니다.\n"
    "문자를 분석하여 반드시 아래 JSON 형식으로만 답하세요.\n"
    '{"verdict": "smishing|suspicious|normal", "risk_factors": [...], "explanation": "..."}\n'
    "- verdict: smishing(사기), suspicious(의심), normal(정상) 중 하나\n"
    "- risk_factors: 발견된 위험 요인 태그 배열 (정상이면 빈 배열)\n"
    "- explanation: 판단 근거를 2~3문장으로, 사용자가 앞으로 유사한 문자를 "
    "스스로 판별할 수 있도록 설명"
)


def build_user_prompt(text: str) -> str:
    return f"다음 문자 메시지를 분석해 주세요.\n\n문자: {text}"


def build_messages(text: str) -> list[dict[str, str]]:
    """OpenAI/HF chat 형식의 messages 리스트를 반환."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text)},
    ]


# ── 설명 라벨 생성 (SFT 타깃 구축용) ──────────────────────────
# 정답 verdict 를 알려주고 근거(risk_factors + explanation)만 생성하게 하는
# label-conditioned 방식. 정답을 알기에 zero-shot 보다 품질이 높다.
# 설계 근거: docs/스키마_프롬프트_설계.md §1, §3

_TAG_GUIDE = (
    "url_shortener(단축 URL), suspicious_domain(공식 아닌/오타 도메인), "
    "impersonation_delivery(택배 사칭), impersonation_gov(정부·수사기관·건강보험 사칭), "
    "impersonation_financial(금융사 사칭), impersonation_acquaintance(지인·가족·부고 사칭), "
    "urgency_pressure(시간 압박), fear_appeal(불안·공포 조성), financial_lure(금전 유인), "
    "app_install_request(앱 설치 유도), personal_info_request(개인정보·인증번호 요구), "
    "callback_induce(회신 전화 유도), abnormal_format(자모 분리 등 필터 우회), "
    "unverifiable_sender(국제발신·발신자 확인 불가), link_only_message(설명 없이 링크만)"
)

LABEL_GEN_SYSTEM = (
    "당신은 한국어 스미싱 판정 학습 데이터의 라벨을 만드는 전문가입니다.\n"
    "문자와 '정답 판정'이 주어집니다. 정답에 맞춰 근거를 생성하세요.\n"
    "risk_factors 는 아래 태그 목록에서만 고르세요(정상이면 빈 배열):\n"
    f"{_TAG_GUIDE}\n"
    "explanation 은 2~3문장으로, ① 이 문자의 어떤 부분이 왜 위험/안전한지 "
    "구체적으로 짚고 ② 사용자가 앞으로 스스로 판별할 수 있는 일반 원칙 1개를 담으세요.\n"
    "반드시 아래 JSON 형식으로만 답하세요.\n"
    '{"verdict": "...", "risk_factors": [...], "explanation": "..."}'
)


def build_label_gen_messages(text: str, verdict: str) -> list[dict[str, str]]:
    """정답 verdict 를 조건으로 근거 라벨을 생성하기 위한 messages."""
    user = (
        f"문자: {text}\n\n"
        f"정답 판정(verdict): {verdict}\n\n"
        "위 정답에 맞는 risk_factors 와 explanation 을 생성해 JSON 으로 답하세요."
    )
    return [
        {"role": "system", "content": LABEL_GEN_SYSTEM},
        {"role": "user", "content": user},
    ]
