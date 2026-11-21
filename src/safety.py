"""Find and hide secrets a customer should never type into a chat (card numbers, PINs, codes).
Done in code, not by the LLM: it must work every time, and the secret must never reach the LLM."""
import re

CARD_NUMBER = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
SECRET_WORDS = re.compile(
    r"\b(pin|passcode|password|otp|cvv|code)\b|السري|\bسري\b|الرمز|\bرمز|كود|كلمة السر|باسورد",
    re.IGNORECASE)
SHORT_NUMBER = re.compile(r"(?<!\d)\d{3,6}(?!\d)")


def mask_sensitive(text: str) -> tuple[str, bool]:
    """Returns (text with secrets hidden, whether anything was hidden)."""
    found = False
    if CARD_NUMBER.search(text):
        text = CARD_NUMBER.sub("[hidden card number]", text)
        found = True
    if SECRET_WORDS.search(text) and SHORT_NUMBER.search(text):
        text = SHORT_NUMBER.sub("[hidden]", text)
        found = True
    return text, found
