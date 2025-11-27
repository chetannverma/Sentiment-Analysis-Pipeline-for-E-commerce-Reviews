# 🧠 Sentiment Analysis Pipeline (FastAPI + ML)

A complete ML pipeline for training, evaluating, and serving a sentiment classifier using:
- Kaggle Amazon Reviews Dataset
- TF-IDF Vectorizer
- Logistic Regression
- FastAPI Deployment API

## 🚀 How to run

### 1. Create environment
pip install -r requirements.txt

### 2. Train the model
python src/train.py

### 3. Start the API
uvicorn src.api:app --reload --port 8000

### 4. Test the API
POST http://localhost:8000/predict
{
  "text": "Amazing phone!"
}

...
