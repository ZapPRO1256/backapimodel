import uvicorn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from deep_translator import GoogleTranslator
from tensorflow.keras.models import load_model
import pickle

# Завантажуємо модель, векторизатор та енкодер
model = load_model('keyword_article_model.keras')
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # можна обмежити домени
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

def preprocess_query(query: str) -> np.ndarray:
    try:
        # Перекладемо запит англійською
        translated = GoogleTranslator(source='auto', target='en').translate(query)
    except Exception as e:
        print(f"Помилка перекладу: {e}")
        translated = query
    # Простіше приведення рядка: видаляємо коми, приводимо до нижнього регістру
    processed = " ".join(translated.lower().replace(",", " ").split())
    # Подвоюємо рядок як у тренуванні
    repeated = f"{processed} {processed}"
    # Векторизуємо
    X = vectorizer.transform([repeated]).toarray()
    return X

def predict(query: str, top_n=5):
    X = preprocess_query(query)
    preds = model.predict(X)[0]
    top_indices = np.argsort(preds)[-top_n:][::-1]
    top_scores = preds[top_indices]
    top_labels = label_encoder.inverse_transform(top_indices)
    results = [{"code": label, "confidence": float(score)} for label, score in zip(top_labels, top_scores)]
    return results

@app.post("/predict")
def get_predictions(request: QueryRequest):
    print(f"Отримано запит: {request.query}")
    predictions = predict(request.query)
    print(f"Прогноз: {predictions}")
    return {"predictions": predictions}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
