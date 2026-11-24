"""Unit tests: fast, no LLM, no internet. Run with:  python -m pytest tests -q"""
from src.safety import mask_sensitive


def test_pin_is_hidden():
    text, found = mask_sensitive("My PIN is 4821, why is my card blocked?")
    assert found and "4821" not in text


def test_arabic_pin_is_hidden():
    text, found = mask_sensitive("رقمي السري 1234")
    assert found and "1234" not in text


def test_arabic_digits_are_hidden():
    text, found = mask_sensitive("الرقم السري ٤٥٦٧")
    assert found and "٤٥٦٧" not in text


def test_card_number_is_hidden():
    text, found = mask_sensitive("my card is 4111 1111 1111 1111")
    assert found and "4111" not in text


def test_amounts_are_not_hidden():
    text, found = mask_sensitive("What is the fee to withdraw 5000 EGP?")
    assert not found and "5000" in text


def test_fast_word_is_not_a_secret():
    text, found = mask_sensitive("تحويل سريع 5000 جنيه")      # "سريع" contains "سري"
    assert not found
