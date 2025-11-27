# src/train.py
import pandas as pd
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def run(input_csv="data/processed_reviews.csv", model_out="models/baseline_tfidf_lr.pkl"):
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["text","label"])
    X = df["text"].astype(str)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test,preds))
    print(classification_report(y_test, preds, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test,preds))

    joblib.dump({"model": clf, "vectorizer": tfidf}, model_out)
    print("Saved model bundle to", model_out)

if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv)>1 else "data/processed_reviews.csv"
    outp = sys.argv[2] if len(sys.argv)>2 else "models/baseline_tfidf_lr.pkl"
    run(input_csv, outp)
