import uvicorn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
import pickle

# ✅ Завантаження моделі SBERT
model = SentenceTransformer("MrZaper/LiteModel")

# ✅ Завантаження векторів статей та відповідних ключів
article_embeddings = np.load('./sbert_embeddings.npy')
with open('./sbert_labels.pkl', "rb") as f:
    article_keys = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

def preprocess_query(query: str) -> str:
    try:
        return GoogleTranslator(source="auto", target="en").translate(query)
    except Exception as e:
        print(f"Помилка перекладу: {e}")
        return query

def predict(query: str, top_n=5):
    translated_query = preprocess_query(query)
    query_embedding = model.encode(translated_query, convert_to_tensor=True)

    # ✅ Обчислення косинусної подібності
    similarities = util.cos_sim(query_embedding, article_embeddings)[0]
    top_results = np.argsort(-similarities.cpu().numpy())[:top_n]

    results = [
        {
            "code": article_keys[idx],
            "confidence": float(similarities[idx])
        }
        for idx in top_results
    ]
    return results

@app.post("/predict")
def get_predictions(request: QueryRequest):
    print(f"Отримано запит: {request.query}")
    predictions = predict(request.query)
    print(f"Прогноз: {predictions}")
    return {"predictions": predictions}

# Якщо хочеш запускати напряму:
# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
