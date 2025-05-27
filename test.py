import os
print("Вміст папки:", os.listdir("."))
print("Розмір keyword_article_model.keras:", os.path.getsize("keyword_article_model.keras"))

from keras.models import load_model
model = load_model("keyword_article_model.keras")
