"""Text cleaning helpers used across the project.

Notebooks, training and the final chatbot all import from here, so the
SAME cleaning runs everywhere. If training used one cleaning and the live
bot used another, the model would see text it was never trained on.

The customer's original message is never changed: these functions make
an internal copy for the model, or a key for finding duplicates.
"""

import re

# Arabic diacritics (tashkeel) such as fatha, damma, kasra, shadda, sukun,
# plus the small superscript alef. Customers rarely type them consistently.
_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰ]")
_TATWEEL = "ـ"                                  # the stretching line in مرتيـــن
_ALEF_FORMS = re.compile(r"[إأآٱ]")                 # all alef shapes -> ا
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# 3 or more of the same ARABIC LETTER in a row (ساعدوووووني) -> one letter.
# - Only Arabic letters (ء-ي): digits are never touched (5000 stays
#   5000), and neither is punctuation or English text.
# - 3+, not 2+: some words really have a double letter (ممكن, اللي), and
#   collapsing pairs would break them.
_REPEATS = re.compile(r"([ء-ي])\1{2,}")


def normalize_arabic(text: str, alef_maqsura: bool = False, taa_marbuta: bool = False) -> str:
    """Make Arabic text consistent before the model sees it.

    Always: remove diacritics and tatweel, unify alef shapes (أ إ آ -> ا),
    Arabic digits -> 0-9, and shorten stretched letters.
    Arabic punctuation (؟ ،) is kept as typed.

    Optional switches, OFF by default because they can change meaning:
    - alef_maqsura: ى -> ي   (على "on" would become علي "Ali")
    - taa_marbuta:  ة -> ه   (barely shrank our vocabulary, see Lesson 4)
    We test them later as experiments and keep them only if they help.
    """
    text = _DIACRITICS.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = _ALEF_FORMS.sub("ا", text)
    if alef_maqsura:
        text = text.replace("ى", "ي")
    if taa_marbuta:
        text = text.replace("ة", "ه")
    text = text.translate(_ARABIC_DIGITS)             # ٥٠٠٠ -> 5000
    text = _REPEATS.sub(r"\1", text)                  # ساعدوووووني -> ساعدوني
    text = re.sub(r"\s+", " ", text).strip()
    return text


def comparison_key(text: str) -> str:
    """A cleaned version of a sentence, used ONLY to find duplicates.

    Lowercase, Arabic normalized (including ى -> ي), no punctuation, single
    spaces. Two sentences with the same key count as the same sentence.
    Here we merge ى and ي on purpose: when hunting for leaks it is safer to
    treat على and علي as the same than to miss a near-copy.
    """
    text = normalize_arabic(text.lower(), alef_maqsura=True)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()