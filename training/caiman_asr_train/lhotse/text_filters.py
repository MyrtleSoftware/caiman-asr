from beartype import beartype
from beartype.typing import List

_BANNED_TRANSCRIPTS = {
    "fr": ["Merci, M. le Président. <EOS>"],
    "es": [
        "Gracias, señora presidenta. <EOS>",
        "Gracias, señor presidente. <EOS>",
        "Muchas gracias.",
        "Gracias.",
        "¡Gracias!",
    ],
    "de": ["Vielen Dank."],
    "it": [
        "Grazie a tutti. <EOS>",
        "Grazie mille. <EOS>",
        "Grazie Presidente. <EOS>",
        "Grazie.",
    ],
    "pt": [
        "Obrigado, Sr. Presidente. <EOS>",
        "Obrigada, Sr. Presidente. <EOS>",
        "Obrigado.",
        "Obrigada.",
    ],
}


@beartype
def get_banned_transcripts(lang_code: str) -> List[str]:
    return _BANNED_TRANSCRIPTS.get(lang_code, [])
