from tensorflow.keras.models import load_model
model = load_model('keyword_article_model.keras')
model.summary()

import os
print("Файли в поточній директорії:", os.listdir("."))
