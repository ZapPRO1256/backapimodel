import uvicorn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
import pickle

model = SentenceTransformer("MrZaper/LiteModel")

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
    similarities = util.cos_sim(query_embedding, article_embeddings)[0].cpu().numpy()
    key_to_best_similarity = {}
    for idx, sim in enumerate(similarities):
        key = article_keys[idx]
        if key not in key_to_best_similarity or sim > key_to_best_similarity[key]:
            key_to_best_similarity[key] = sim
    sorted_results = sorted(
        key_to_best_similarity.items(),
        key=lambda x: x[1],
        reverse=True
    )
    results = [
        {
            "code": key,
            "confidence": float(similarity)
        }
        for key, similarity in sorted_results[:top_n]
    ]
    return results

@app.post("/predict")
def get_predictions(request: QueryRequest):
    print(f"Отримано запит: {request.query}")
    predictions = predict(request.query)
    print(f"Прогноз: {predictions}")
    return {"predictions": predictions}
