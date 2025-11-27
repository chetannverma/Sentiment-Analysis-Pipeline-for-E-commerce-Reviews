# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from typing import List

MODEL_PATH = "models/baseline_tfidf_lr.pkl"

# Load model bundle
bundle = joblib.load(MODEL_PATH)
clf = bundle["model"]
tfidf = bundle["vectorizer"]

app = FastAPI(title="Ecom Sentiment API")

class SingleIn(BaseModel):
    text: str

class BatchIn(BaseModel):
    texts: List[str]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(payload: SingleIn):
    txt = payload.text
    vec = tfidf.transform([txt])
    pred = int(clf.predict(vec)[0])
    proba = clf.predict_proba(vec).tolist()[0] if hasattr(clf, "predict_proba") else []
    return {"text": txt, "label": pred, "proba": proba}

@app.post("/batch_predict")
def batch_predict(payload: BatchIn):
    vecs = tfidf.transform(payload.texts)
    preds = clf.predict(vecs).tolist()
    probas = clf.predict_proba(vecs).tolist() if hasattr(clf, "predict_proba") else [[]] * len(preds)
    return {
        "results": [
            {"text": t, "label": int(p), "proba": pr}
            for t, p, pr in zip(payload.texts, preds, probas)
        ]
    }

# Only run local server when executing `python src/api.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
