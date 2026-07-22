import re

PATROES_SUSPEITOS = [
    r"ignore\s+(all\s+|the\s+)?(previous|above)\s+instructions",
    r"revel(e|ar|es|em)?\s+seu\s+prompt",
    r"você\s+agora\s+é\s+um",
    r"system\s+prompt",
    r"sudo\s+mode"
]

def check_input_guardrail(user_input: str) -> str:
    """
    Validates the user input against prompt injection patterns and size limits.
    """
    if not user_input or not user_input.strip():
        raise ValueError("A pergunta não pode estar vazia.")

    # 1. RegEx check for known injection terms
    for pattern in PATROES_SUSPEITOS:
        if re.search(pattern, user_input, re.IGNORECASE):
            raise ValueError("Entrada inválida detectada por motivos de segurança.")

    # 2. Structural limit
    if len(user_input.strip()) > 1000:
        raise ValueError("A pergunta excede o tamanho máximo permitido.")

    return user_input
