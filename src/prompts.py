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
