# %% [markdown]
# # Lesson 4: How a model reads Arabic (normalization and tokenization)
#
# Two questions:
# 1. Which Arabic spelling differences should we remove before the model reads the text?
# 2. How does each candidate model cut a sentence into pieces (tokens), and
#    which one handles English, MSA and Egyptian Arabic best?
#
# Setup (once, in the terminal with (.venv) active):
#   pip install transformers
# Run notebook 02 first: this notebook reads ../data/banking_en_ar.csv

# %%
import re
import sys
from collections import Counter

import pandas as pd

sys.path.append("..")
from src.text_utils import normalize_arabic

data = pd.read_csv("../data/banking_en_ar.csv")
ar = data[data["lang"] == "ar"]
en = data[data["lang"] == "en"]
print(len(ar), "Arabic rows |", len(en), "English rows")

# %% [markdown]
# ## 1. What normalization changes
# Each example shows one problem that real customers create.

# %%
examples = [
    "أريد تحويل ٥٠٠٠ جنيه إلى حسابي؟",     # hamza alef, Arabic digits (5000 must stay 5000)
    "عَايِز أَغَيَّر الرَّقَم السِّرِّي",          # diacritics (tashkeel)
    "الكارت اتسرق ساعدوووووني!!!",          # stretched letters (punctuation is kept)
    "الفلوس اتسحبت مرتيـــــن",              # tatweel (the long line)
    "ممكن اعرف اللي حصل",                    # real double letters must NOT be shortened
    "عايز اعرف الرصيد على الفيزا",           # على stays على (ى -> ي is off by default)
]
for text in examples:
    print("before:", text)
    print("after: ", normalize_arabic(text))
    print()

# The optional switch, for comparison: على becomes علي ("on" becomes "Ali")
print("with alef_maqsura=True:", normalize_arabic(examples[-1], alef_maqsura=True))

# %% [markdown]
# ## 2. How often does each problem appear in our Arabic data?
# Our data is translated MSA, so it is fairly clean. Real customer messages
# (and our Egyptian data later) will have much more of these.

# %%
checks = {
    "diacritics":            r"[ً-ْ]",
    "tatweel (ـ)":           "ـ",
    "hamza alef (أ إ آ)":    "[أإآ]",
    "alef maqsura (ى)":      "ى",
    "Arabic digits (٠-٩)":   "[٠-٩]",
}
for name, pattern in checks.items():
    count = ar["text"].str.contains(pattern, regex=True).sum()
    print(f"{name:22} {count:6} rows  ({count / len(ar):.0%})")

# %% [markdown]
# ## 3. Does normalization help? Count the vocabulary.
# Vocabulary = the set of different words. If normalization merges spellings
# of the same word, the vocabulary gets smaller and each word gets more
# examples to learn from.
#
# We also measure the two optional switches (ى -> ي and ة -> ه). They are
# off by default because they can change meaning; here we only see how
# much they would merge. Whether they help the MODEL is tested later.


# %%
def words(text):
    """Split into words, ignoring punctuation (so اهلا! and اهلا count as one word)."""
    return re.findall(r"\w+", text)


def vocab(texts):
    return {word for text in texts for word in words(text)}


train_ar = ar.loc[ar["split"] == "train", "text"]
v_raw = vocab(train_ar)
v_norm = vocab(train_ar.map(normalize_arabic))
v_alef = vocab(train_ar.map(lambda t: normalize_arabic(t, alef_maqsura=True)))
v_taa = vocab(train_ar.map(lambda t: normalize_arabic(t, taa_marbuta=True)))

print("unique words, raw:                ", len(v_raw))
print("unique words, normalized:         ", len(v_norm))
print("  + switch ى -> ي (not used yet): ", len(v_alef))
print("  + switch ة -> ه (not used yet): ", len(v_taa))

# %% [markdown]
# ## 4. Why we cannot just use whole words: unseen words
# A word-level model only knows words it saw in training. Any new word is
# "out of vocabulary" (OOV) and the model is blind to it.
# How many words in the Arabic TEST set never appear in training?

# %%
test_words = [w for t in ar.loc[ar["split"] == "test", "text"].map(normalize_arabic) for w in words(t)]
oov = [w for w in test_words if w not in v_norm]
print(f"{len(oov)} of {len(test_words)} test words are unseen ({len(oov) / len(test_words):.1%})")
print("examples:", Counter(oov).most_common(10))

# %% [markdown]
# ## 5. Subword tokenization: what real models do
# Models cut text into SUBWORDS: common words stay whole, rare words are
# split into known pieces, so (almost) nothing is unseen.
#
# We compare three candidate models:
# - MARBERTv2: Arabic only, trained on a billion Arabic tweets (lots of dialect)
# - CAMeLBERT-mix: Arabic only, trained on MSA + dialects + classical Arabic
# - XLM-RoBERTa: multilingual, 100 languages including Arabic and English
#
# The first run downloads each tokenizer (a few MB each) from Hugging Face.
# A warning that PyTorch is not installed is fine: we only need tokenizers today.

# %%
from transformers import AutoTokenizer

MODELS = {
    "MARBERTv2": "UBC-NLP/MARBERTv2",
    "CAMeLBERT-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "XLM-RoBERTa": "FacebookAI/xlm-roberta-base",
}
tokenizers = {name: AutoTokenizer.from_pretrained(model_id) for name, model_id in MODELS.items()}

for name, tok in tokenizers.items():
    print(f"{name:14} vocabulary size: {tok.vocab_size:,}")

# %% [markdown]
# ## 6. See the pieces
# `##` (BERT models) = this piece continues the previous one.
# `▁` (XLM-RoBERTa) = this piece starts a new word.
# `[UNK]` / `<unk>` = unknown: the model cannot read this piece at all.

# %%
sentences = {
    "English":  "My card was stolen, please block it",
    "MSA":      "لقد سُرقت بطاقتي، أرجو إيقافها",
    "Egyptian": "الكارت بتاعي اتسرق، عايز أوقفه",
    "Franco":   "el card bta3ty et2aslet 3ayez awa2afha",
    "Mixed":    "عايز أعمل transfer بس الـ app مش شغال",
}
for style, text in sentences.items():
    print(f"=== {style}: {text}")
    for name, tok in tokenizers.items():
        pieces = tok.tokenize(normalize_arabic(text))
        print(f"  {name:14} ({len(pieces):2} tokens) {pieces}")
    print()

# %% [markdown]
# ## 7. Measure it: fertility and unknown tokens
# - Fertility = tokens per word. Lower usually means the model knows the
#   language better (it keeps words whole). Around 1.2-1.5 is good; 3+ means
#   it is chopping words into fragments.
# - Unknown rate = share of tokens the model cannot read. Should be ~0%.
#
# We measure on 1,000 random English and 1,000 MSA test questions, plus a
# few Egyptian and Franco sentences I wrote as a preview. These are NOT our
# gold test set: they are only here to show the trend.

# %%
egyptian_preview = [
    "الفيزا بتاعتي ضاعت ومش لاقيها",
    "عايز أعرف رصيدي كام",
    "الفلوس اتسحبت مرتين من حسابي",
    "ليه التحويل اتأخر كده",
    "المكنة بلعت الكارت بتاعي",
    "عايز أغير الرقم السري بتاع الفيزا",
]
franco_preview = [
    "el visa bta3ty da3et",
    "3ayez a3raf rasidi kam",
    "el felous etsa7bet marteen",
    "leh el ta7weel et2a5ar keda",
]

groups = {
    "English": en.loc[en["split"] == "test", "text"].sample(1000, random_state=0).tolist(),
    "MSA": ar.loc[ar["split"] == "test", "text"].sample(1000, random_state=0).tolist(),
    "Egyptian (preview)": egyptian_preview,
    "Franco (preview)": franco_preview,
}


def measure(tok, texts):
    words = tokens = unknown = 0
    for text in texts:
        text = normalize_arabic(text)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        words += len(text.split())
        tokens += len(ids)
        unknown += sum(i == tok.unk_token_id for i in ids)
    return round(tokens / words, 2), round(100 * unknown / tokens, 2)


rows = []
for name, tok in tokenizers.items():
    for group, texts in groups.items():
        fertility, unk = measure(tok, texts)
        rows.append({"model": name, "text": group, "fertility": fertility, "unknown %": unk})

results = pd.DataFrame(rows)
results.pivot(index="model", columns="text", values="fertility")

# %%
results.pivot(index="model", columns="text", values="unknown %")

# %% [markdown]
# ## 8. Special tokens and length
# Models add special tokens around every input:
# - BERT: [CLS] at the start (its final vector represents the whole sentence,
#   used for classification) and [SEP] at the end.
# - XLM-RoBERTa: <s> and </s> do the same job.
# Models also have a maximum length (512 tokens for these). Our questions
# are much shorter, so we will use a smaller limit later to train faster.

# %%
for name, tok in tokenizers.items():
    ids = tok("عايز أوقف الكارت")["input_ids"]
    print(f"{name:14}", tok.convert_ids_to_tokens(ids))
# %%
