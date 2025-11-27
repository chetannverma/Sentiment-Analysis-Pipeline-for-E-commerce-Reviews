# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import sys
from pathlib import Path

def run(input_csv="data/processed_reviews.csv", out_dir="reports"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv, parse_dates=["review_date"], dayfirst=True, infer_datetime_format=True)

    # Basic stats
    total = len(df)
    mean_rating = df["rating"].mean()
    median_rating = df["rating"].median()
    counts = df["label"].value_counts()
    print(f"Total reviews: {total}, mean rating: {mean_rating}, median: {median_rating}")
    print("Sentiment counts:\n", counts)

    # Word count distribution
    plt.figure(figsize=(6,4))
    sns.histplot(df["word_count"], bins=30)
    plt.title("Word count distribution")
    plt.savefig(f"{out_dir}/wordcount_dist.png", bbox_inches="tight")

    # Sentiment distribution pie
    plt.figure(figsize=(4,4))
    counts.plot(kind="pie", autopct="%1.1f%%", labels=["Positive","Negative"] if 1 in counts.index else counts.index)
    plt.title("Sentiment distribution")
    plt.savefig(f"{out_dir}/sentiment_pie.png", bbox_inches="tight")

    # Monthly sentiment trend
    if "review_date" in df.columns and not df["review_date"].isnull().all():
        df["month"] = pd.to_datetime(df["review_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        trend = df.groupby("month")["label"].mean().reset_index()
        plt.figure(figsize=(8,4))
        sns.lineplot(data=trend, x="month", y="label", marker="o")
        plt.title("Monthly Average Sentiment (1=positive,0=negative)")
        plt.savefig(f"{out_dir}/monthly_trend.png", bbox_inches="tight")
    else:
        print("No date column present or entirely null; skipping monthly trend.")

    # Wordclouds: negative and positive
    pos_text = " ".join(df[df["label"]==1]["text"].astype(str).tolist())
    neg_text = " ".join(df[df["label"]==0]["text"].astype(str).tolist())

    wc_pos = WordCloud(width=800, height=400).generate(pos_text if pos_text.strip() else "good")
    wc_neg = WordCloud(width=800, height=400).generate(neg_text if neg_text.strip() else "bad")

    wc_pos.to_file(f"{out_dir}/wc_positive.png")
    wc_neg.to_file(f"{out_dir}/wc_negative.png")

    # Top pain points (simple frequency of tokens in negative reviews)
    from collections import Counter
    tokens = " ".join(df[df["label"]==0]["text"].astype(str).tolist()).split()
    top = Counter(tokens).most_common(30)
    top_df = pd.DataFrame(top, columns=["term","count"])
    top_df.to_csv(f"{out_dir}/top_negative_terms.csv", index=False)

    print("EDA artifacts saved to", out_dir)
    return

if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv)>1 else "data/processed_reviews.csv"
    run(input_csv)
