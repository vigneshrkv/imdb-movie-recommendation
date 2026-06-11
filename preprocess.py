"""
preprocess.py
-------------
Text Preprocessing Script for IMDb 2024 Movie Storylines.

INPUT  : imdb_movies_raw.csv        (2 columns  → Movie Name, Storyline)
OUTPUT : imdb_movies_2024.csv       (3 columns  → Movie Name, Storyline, Cleaned_Storyline)

Cleaning Steps Applied:
    Step 1 → Lowercase conversion
    Step 2 → Remove punctuation
    Step 3 → Remove numbers
    Step 4 → Remove extra whitespace
    Step 5 → Remove stopwords (NLTK English stopwords)
    Step 6 → Remove short tokens (length ≤ 1)

Run with:
    python preprocess.py
"""

import re
import pandas as pd
import nltk

from nltk.corpus import stopwords

# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD NLTK STOPWORDS
# ─────────────────────────────────────────────────────────────────────────────

nltk.download("stopwords", quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE  = "imdb_movies_raw.csv"       # CSV with 2 columns (Movie Name, Storyline)
OUTPUT_FILE = "imdb_movies_2024.csv"      # CSV with 3 columns (adds Cleaned_Storyline)

# ─────────────────────────────────────────────────────────────────────────────
# STOPWORDS
# ─────────────────────────────────────────────────────────────────────────────

STOP_WORDS = set(stopwords.words("english"))

# ─────────────────────────────────────────────────────────────────────────────
# STEP-BY-STEP CLEANING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def step1_lowercase(text: str) -> str:
    """
    Step 1: Convert all characters to lowercase.
    Example: "A Fading CELEBRITY" → "a fading celebrity"
    """
    return text.lower()


def step2_remove_punctuation(text: str) -> str:
    """
    Step 2: Remove all punctuation characters.
    Example: "black-market drug: a cell-replicating" → "blackmarket drug  a cellreplicating"
    """
    return re.sub(r"[^\w\s]", " ", text)


def step3_remove_numbers(text: str) -> str:
    """
    Step 3: Remove all numeric digits.
    Example: "3 friends travel 2024" → "  friends travel     "
    """
    return re.sub(r"\d+", " ", text)


def step4_remove_extra_whitespace(text: str) -> str:
    """
    Step 4: Strip leading/trailing spaces and collapse multiple spaces into one.
    Example: "  fading   celebrity  " → "fading celebrity"
    """
    return re.sub(r"\s+", " ", text).strip()


def step5_remove_stopwords(text: str) -> str:
    """
    Step 5: Remove common English stopwords (NLTK stopword list).
    Example: "a fading celebrity takes a drug" → "fading celebrity takes drug"
    """
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(tokens)


def step6_remove_short_tokens(text: str) -> str:
    """
    Step 6: Remove tokens with length ≤ 1 (single characters left after cleaning).
    Example: "fading celebrity a b takes" → "fading celebrity takes"
    """
    tokens = text.split()
    tokens = [word for word in tokens if len(word) > 1]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# MASTER CLEANING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def clean_storyline(text: str) -> str:
    """
    Apply all 6 cleaning steps in sequence to a single storyline string.

    Pipeline:
        Raw Text
           ↓ Step 1 — Lowercase
           ↓ Step 2 — Remove Punctuation
           ↓ Step 3 — Remove Numbers
           ↓ Step 4 — Remove Extra Whitespace
           ↓ Step 5 — Remove Stopwords
           ↓ Step 6 — Remove Short Tokens
        Cleaned Text
    """
    text = step1_lowercase(text)
    text = step2_remove_punctuation(text)
    text = step3_remove_numbers(text)
    text = step4_remove_extra_whitespace(text)
    text = step5_remove_stopwords(text)
    text = step6_remove_short_tokens(text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── Load raw CSV (2 columns) ──────────────────────────────────────────────
    print("=" * 55)
    print("  IMDb Storyline Preprocessing Pipeline")
    print("=" * 55)

    print(f"\n📂  Loading  : {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    print(f"✅  Loaded   : {len(df):,} movies")
    print(f"📋  Columns  : {list(df.columns)}")

    # Validate required columns
    if "Movie Name" not in df.columns or "Storyline" not in df.columns:
        raise ValueError(
            "CSV must have 'Movie Name' and 'Storyline' columns."
        )

    # Drop rows where Storyline is missing
    before = len(df)
    df.dropna(subset=["Storyline"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    after = len(df)

    if before != after:
        print(f"⚠️   Dropped {before - after} rows with missing storylines.")

    # ── Apply Cleaning Pipeline ───────────────────────────────────────────────
    print("\n🔄  Applying cleaning pipeline...")
    print("    Step 1 → Lowercase conversion")
    print("    Step 2 → Remove punctuation")
    print("    Step 3 → Remove numbers")
    print("    Step 4 → Remove extra whitespace")
    print("    Step 5 → Remove stopwords")
    print("    Step 6 → Remove short tokens")

    df["Cleaned_Storyline"] = df["Storyline"].apply(clean_storyline)

    print(f"\n✅  Cleaning complete!")

    # ── Show Before / After Sample ────────────────────────────────────────────
    print("\n" + "-" * 55)
    print("  Sample: Before vs After Cleaning")
    print("-" * 55)
    for i in range(min(3, len(df))):
        print(f"\n  Movie   : {df['Movie Name'][i]}")
        print(f"  Before  : {df['Storyline'][i]}")
        print(f"  After   : {df['Cleaned_Storyline'][i]}")

    # ── Save Output CSV (3 columns) ───────────────────────────────────────────
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\n" + "=" * 55)
    print(f"  📁  Saved to  : {OUTPUT_FILE}")
    print(f"  📊  Rows      : {len(df):,}")
    print(f"  📋  Columns   : {list(df.columns)}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
