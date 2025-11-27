# src/dashboard.py
import streamlit as st
import pandas as pd
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt

@st.cache_data
def load_data(path="data/processed_reviews.csv"):
    return pd.read_csv(path, parse_dates=["review_date"], dayfirst=True, infer_datetime_format=True)

df = load_data()

st.title("Sentiment Intelligence Dashboard — Amazon Unlocked Phones")
st.metric("Total reviews", len(df))
st.metric("Mean rating", round(df["rating"].mean(),2))
st.metric("Median rating", round(df["rating"].median(),2))

st.header("Sentiment distribution")
st.write(df["label"].value_counts().rename({1:"Positive",0:"Negative"}))

st.header("Monthly sentiment trend")
if "review_date" in df.columns and not df["review_date"].isnull().all():
    df["month"] = pd.to_datetime(df["review_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    trend = df.groupby("month")["label"].mean().reset_index()
    chart = alt.Chart(trend).mark_line(point=True).encode(x="month:T", y="label:Q")
    st.altair_chart(chart, use_container_width=True)
else:
    st.write("No valid date column available")

st.header("Top negative terms (simple)")
neg = " ".join(df[df["label"]==0]["text"].astype(str).tolist())
from collections import Counter
terms = Counter(neg.split())
top = terms.most_common(20)
st.bar_chart(pd.DataFrame(top, columns=["term","count"]).set_index("term"))

st.header("Sample negative reviews")
for r in df[df["label"]==0]["text"].sample(5).tolist():
    st.write("-", r[:300])
