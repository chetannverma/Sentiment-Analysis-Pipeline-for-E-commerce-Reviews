# src/preprocess.py
import os
import pandas as pd
import re
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

STOP = set(stopwords.words("english"))
LEMMA = WordNetLemmatizer()

# --------------------------------------------------------
# CLEANING FUNCTIONS
# --------------------------------------------------------

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    toks = [LEMMA.lemmatize(t) for t in s.split() if t not in STOP and len(t) > 1]
    return " ".join(toks)

def map_rating_to_binary(r):
    try:
        r = float(r)
    except:
        return None
    if r >= 4.0:
        return 1
    if r <= 2.0:
        return 0
    return None     # ignore 3-star reviews

# --------------------------------------------------------
# MAIN
# --------------------------------------------------------

def main(input_path="data/amazon_reviews_unlocked_phones.csv",
         output_path="data/processed_reviews.csv",
         sample_products=None):

    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)

    print("📥 Loading dataset:", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    print("➡ Shape:", df.shape)

    # ----------------------------------------------------
    # MANUAL COLUMN MAPPING for Kaggle dataset
    # ----------------------------------------------------
    print("\n🔍 Overriding column detection for Kaggle dataset...\n")

    text_col = "Reviews"
    rating_col = "Rating"
    product_col = "Product Name"

    # Validate presence
    for col in [text_col, rating_col, product_col]:
        if col not in df.columns:
            raise ValueError(f"❌ Expected column '{col}' not found in CSV.")

    print("Detected columns:")
    print("   text      → Reviews")
    print("   rating    → Rating")
    print("   product   → Product Name")
    print()

    # Filter products if needed
    if sample_products:
        df = df[df[product_col].isin(sample_products)]
        print("➡ Filtered to sample products. New shape:", df.shape)

    # ----------------------------------------------------
    # Standardize fields
    # ----------------------------------------------------
    df["review_text_raw"] = df[text_col].astype(str)
    df["rating_raw"] = df[rating_col]
    df["product_raw"] = df[product_col]

    df["review_date_raw"] = None
    df["verified_raw"] = None

    # Remove NA
    df = df[df["review_text_raw"].notna() & df["rating_raw"].notna()].copy()

    # Binary sentiment
    df["label"] = df["rating_raw"].apply(map_rating_to_binary)
    df = df[df["label"].notna()].copy()

    print("➡ After filtering neutral/invalid ratings:", df.shape)

    # ----------------------------------------------------
    # Cleaning
    # ----------------------------------------------------
    print("\n🧼 Cleaning text...")

    df["text_clean"] = df["review_text_raw"].apply(clean_text)
    df["title_clean"] = ""
    df["text_full"] = df["text_clean"]

    # Word / char counts
    df["word_count"] = df["text_full"].apply(lambda x: len(str(x).split()))
    df["char_count"] = df["text_full"].apply(lambda x: len(str(x)))

    # ----------------------------------------------------
    # Save output
    # ----------------------------------------------------
    df_out = df[[
        "product_raw",
        "review_date_raw",
        "verified_raw",
        "rating_raw",
        "label",
        "text_full",
        "word_count",
        "char_count",
        "review_text_raw"
    ]].copy()

    df_out.columns = [
        "product",
        "review_date",
        "verified",
        "rating",
        "label",
        "text",
        "word_count",
        "char_count",
        "raw_text"
    ]

    df_out.to_csv(output_path, index=False, encoding="utf-8")

    print("\n💾 Saved processed file to:", output_path)
    print("\n📌 Sample rows:")
    print(df_out.head())

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/amazon_reviews_unlocked_phones.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed_reviews.csv"
    main(input_path=input_path, output_path=output_path)
