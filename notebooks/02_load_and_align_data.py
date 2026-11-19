# %% [markdown]
# # Lesson 3: Load the English and Arabic banking datasets and line them up
#
# By the end of this notebook we have ONE clean table with every question,
# its intent (the same 77 labels for both languages), its language, and
# whether it is for training, validation or testing.
#
# Setup (run once in your terminal, with your virtual environment active):
#   pip install pandas matplotlib scikit-learn
#
# Run each "# %%" cell one at a time (VS Code shows a "Run Cell" button).

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("..")                       # lets Python find our src/ folder
from src.text_utils import comparison_key   # shared cleaning (see Lesson 4)

DATA_DIR = Path("../data")          # where we save our cleaned files
DATA_DIR.mkdir(exist_ok=True)

# Both datasets are public CSV files on GitHub. pandas can read a CSV
# straight from a URL, so there is nothing to download by hand.
EN_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
AR_URL = "https://raw.githubusercontent.com/SinaLab/ArBanking77/main/data/"

# Our customers are Egyptian, so we leave the Palestinian dialect out.
# Set this to True later if you want to test whether it helps (an experiment).
KEEP_PAL = False

# %% [markdown]
# ## 1. Load the English data (Banking77)
# Two files: train (to learn from) and test (to grade on at the very end).
# Columns: `text` = the customer question, `category` = its intent.

# %%
en_train = pd.read_csv(EN_URL + "train.csv")
en_test = pd.read_csv(EN_URL + "test.csv")

print(en_train.shape, en_test.shape)   # (rows, columns)
en_train.sample(5, random_state=1)

# %% [markdown]
# ## 2. Load the Arabic data (ArBanking77)
# - test files are already separate per dialect
# - train and validation files contain MSA first, then Palestinian, in one file
#   with no column saying which is which. We found where the Palestinian part
#   starts (row 10733 in train, row 1230 in validation) by checking the text,
#   and the MSA counts match the numbers published in the dataset paper.
# Columns: `text` = question, `label` = intent written in Arabic.

# %%
ar_train = pd.read_csv(AR_URL + "Banking77_Arabized_MSA_PAL_train.csv")
ar_val = pd.read_csv(AR_URL + "Banking77_Arabized_MSA_PAL_val.csv")
ar_test_msa = pd.read_csv(AR_URL + "Banking77_Arabized_MSA_test.csv")
ar_test_pal = pd.read_csv(AR_URL + "Banking77_Arabized_PAL_test.csv")

PAL_START_TRAIN, PAL_START_VAL = 10733, 1230
ar_train["variety"] = ["msa" if i < PAL_START_TRAIN else "pal" for i in range(len(ar_train))]
ar_val["variety"] = ["msa" if i < PAL_START_VAL else "pal" for i in range(len(ar_val))]

# Look at the rows on both sides of the boundary: MSA (formal) then Palestinian.
for sentence in ar_train["text"].iloc[PAL_START_TRAIN - 3:PAL_START_TRAIN + 3]:
    print(sentence)

if not KEEP_PAL:
    ar_train = ar_train[ar_train["variety"] == "msa"]
    ar_val = ar_val[ar_val["variety"] == "msa"]

print(ar_train.shape, ar_val.shape, ar_test_msa.shape, ar_test_pal.shape)
ar_train.sample(5, random_state=1)

# %% [markdown]
# ## 3. Line up the labels
# English labels look like `card_arrival`; Arabic labels look like `وصول البطاقة`.
# The ArBanking77 authors give us a table mapping one to the other.
# We convert every Arabic label to its English id, so both languages
# share ONE label space.

# %%
intents = pd.read_csv(AR_URL + "Banking77_intents.csv")
intents["intent"] = intents["label_en"].str.strip().str.replace(" ", "_")
ar_to_en = dict(zip(intents["label_ar"].str.strip(), intents["intent"]))

for df in (ar_train, ar_val, ar_test_msa, ar_test_pal):
    df["intent"] = df["label"].str.strip().map(ar_to_en)

# Safety check: every Arabic label must have found an English match.
# If this number is not 0, a label is spelled differently and we must fix it.
print("Unmapped Arabic labels:", sum(df["intent"].isna().sum()
      for df in (ar_train, ar_val, ar_test_msa, ar_test_pal)))
print("English intents:", en_train["category"].nunique(),
      "| Arabic intents:", ar_train["intent"].nunique())

# %% [markdown]
# ## 4. English validation set: chosen later (section 6b)
# Arabic already has train / validation / test. English only has train / test.
# We first used a random 10% of English train as validation, but that LEAKED:
# the Arabic questions are translations of the English ones, so a sentence in
# English val had its Arabic translation in Arabic train (see section 6b).
# So for now all English train rows stay "train"; section 6b picks English
# validation so it matches the Arabic validation questions.

# %%
en_val = en_train.iloc[0:0]   # empty for now, filled in section 6b
print(len(en_train), len(en_val))

# %% [markdown]
# ## 5. Combine everything into one table
# Columns we keep:
# - `text`: the question
# - `intent`: one of the 77 shared labels
# - `lang`: en / ar
# - `variety`: which kind of language (english, msa, pal; later egy)
# - `split`: train / val / test


# %%
def part(df, text_col, intent_col, lang, variety, split):
    return pd.DataFrame({
        "text": df[text_col].astype(str).str.strip(),
        "intent": df[intent_col],
        "lang": lang,
        "variety": variety,
        "split": split,
    })


parts = [
    part(en_train, "text", "category", "en", "english", "train"),
    part(en_val, "text", "category", "en", "english", "val"),
    part(en_test, "text", "category", "en", "english", "test"),
    part(ar_train, "text", "intent", "ar", ar_train["variety"], "train"),
    part(ar_val, "text", "intent", "ar", ar_val["variety"], "val"),
    part(ar_test_msa, "text", "intent", "ar", "msa", "test"),
]
if KEEP_PAL:
    parts.append(part(ar_test_pal, "text", "intent", "ar", "pal", "test"))

data = pd.concat(parts, ignore_index=True)

pd.crosstab(data["variety"], data["split"], margins=True)

# %% [markdown]
# ## 6. Check for leakage and duplicates
# If the same sentence is in train AND val/test, the model can just
# remember it, and the score lies. "Same" includes near-duplicates that
# differ only by a comma, a space, a capital letter or an Arabic spelling
# variant (أ / ا), so we compare a cleaned "key" instead of the raw text.
# comparison_key lives in src/text_utils.py (explained in Lesson 4).
# Leaked rows are removed from train (we never remove test rows to look better).

# %%
data["key"] = data["text"].map(comparison_key)

eval_keys = set(data.loc[data["split"] != "train", "key"])
leak = data["split"].eq("train") & data["key"].isin(eval_keys)
print("Leaked train rows found and removed:", leak.sum())
data = data[~leak]

dupes = data.duplicated(subset=["key", "split"])
print("Duplicate rows found and removed:", dupes.sum())
data = data[~dupes]

# Final check: both numbers must be 0.
eval_keys = set(data.loc[data["split"] != "train", "key"])
print("Leaks left:", (data["split"].eq("train") & data["key"].isin(eval_keys)).sum())
print("Duplicates left:", data.duplicated(subset=["key", "split"]).sum())

# %% [markdown]
# ## 6b. Translation leakage (cross-lingual leakage)
# Every Arabic question was translated from one English Banking77 question.
# A sentence and its translation mean the same thing, so if one is in train
# and the other in val/test, a multilingual model has effectively seen the
# answer. Section 6 can't catch this: the texts are in different languages.
#
# Fix: a **group-aware split**. The "group" of a question is its English
# original. Every member of a group (the English sentence + its Arabic
# translations) must be in the SAME split.
# - Test is kept exactly as published (so we stay comparable with the paper).
# - Arabic train is kept exactly as published (same reason).
# - We drop the few train/val rows whose group is in test,
#   and the Arabic val rows whose sibling translation is in Arabic train.
# - English val = the English originals of the Arabic val questions.
#
# The file `Banking77_full_corpus.csv` gives each English question with its
# Arabic translations side by side, so we can build the groups.
# (A few rows in it have broken quotes; we skip those.)

# %%
full = pd.read_csv(AR_URL + "Banking77_full_corpus.csv", engine="python", on_bad_lines="skip")
english_original = {}
for col in ["Question_MSA1", "Question_MSA2", "Question_PAL1", "Question_PAL2"]:
    for ar, en in zip(full[col], full["Question_en"]):
        if isinstance(ar, str) and ar != "NULL":
            english_original[comparison_key(ar)] = comparison_key(en)

data["group"] = [english_original.get(k, k) if lang == "ar" else k
                 for k, lang in zip(data["key"], data["lang"])]
matched = data.loc[data["lang"] == "ar", "key"].isin(english_original).mean()
print(f"Arabic rows linked to their English original: {matched:.1%}")


def groups_in(split, lang=None):
    rows = data["split"].eq(split)
    if lang:
        rows &= data["lang"].eq(lang)
    return set(data.loc[rows, "group"])


# 1. Nothing outside test may share a group with test.
test_groups = groups_in("test")
bad = data["split"].ne("test") & data["group"].isin(test_groups)
print("Train/val rows removed (translation of a test question):", bad.sum())
data = data[~bad]

# 2. Arabic val rows whose sibling translation is in Arabic train: remove from val.
ar_train_groups = groups_in("train", "ar")
bad = data["split"].eq("val") & data["lang"].eq("ar") & data["group"].isin(ar_train_groups)
print("Arabic val rows removed (sibling translation in train):", bad.sum())
data = data[~bad]

# 3. English validation = English originals of the remaining Arabic val questions.
ar_val_groups = groups_in("val", "ar")
to_val = data["lang"].eq("en") & data["split"].eq("train") & data["group"].isin(ar_val_groups)
data.loc[to_val, "split"] = "val"
print("English rows moved to val:", to_val.sum())

# Final checks: every group in exactly one split, and no exact-text leaks.
splits_per_group = data.groupby("group")["split"].nunique()
print("Groups in more than one split:", (splits_per_group > 1).sum())
eval_keys = set(data.loc[data["split"] != "train", "key"])
print("Leaks left:", (data["split"].eq("train") & data["key"].isin(eval_keys)).sum())
print("Intents missing from English val:",
      77 - data.loc[data["split"].eq("val") & data["lang"].eq("en"), "intent"].nunique())

data = data.drop(columns=["key", "group"]).reset_index(drop=True)
pd.crosstab([data["lang"], data["variety"]], data["split"], margins=True)

# %% [markdown]
# ## 7. Explore the data
# Two questions matter before training:
# 1. Are some intents much more common than others? (class imbalance)
# 2. How long are the questions in each language?

# %%
train = data[data["split"] == "train"]
counts = train["intent"].value_counts()
print("Examples per intent: min", counts.min(), "| max", counts.max())
print("Rarest intents:\n", counts.tail(5))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts.plot.bar(ax=axes[0], title="Training examples per intent")
axes[0].set_xticks([])            # 77 names don't fit, the shape is what matters
axes[0].set_xlabel("77 intents, most to least common")

train.assign(words=train["text"].str.split().str.len()) \
     .boxplot(column="words", by="lang", ax=axes[1])
axes[1].set_title("Words per question")
plt.suptitle("")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Save the clean data and the label list
# `labels.json` fixes the order of the 77 intents. The model will output
# a number (0-76); this file translates that number back into an intent.

# %%
labels = sorted(data["intent"].unique())
label2id = {label: i for i, label in enumerate(labels)}

data.to_csv(DATA_DIR / "banking_en_ar.csv", index=False, encoding="utf-8")
with open(DATA_DIR / "labels.json", "w", encoding="utf-8") as f:
    json.dump(label2id, f, indent=2)

print("Saved", len(data), "rows and", len(labels), "labels")
