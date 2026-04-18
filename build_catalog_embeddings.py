import json
import io
import requests
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array

IMG_SIZE = 224

embedding_model = load_model("final_embedding_model.keras", compile=False)

def load_and_prepare_image_from_url(image_url):
    response = requests.get(image_url, timeout=20)
    response.raise_for_status()

    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    arr = img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    return arr

with open("catalog_source.json", "r", encoding="utf-8") as f:
    catalog_source = json.load(f)

catalog_embeddings = []

for item in catalog_source:
    try:
        x = load_and_prepare_image_from_url(item["imageUrl"])
        emb = embedding_model.predict(x, verbose=0)[0]

        catalog_embeddings.append({
            "productId": item["productId"],
            "category": item["category"],
            "imageUrl": item["imageUrl"],
            "embedding": emb.tolist()
        })

        print("OK:", item["productId"])
    except Exception as e:
        print("SKIP:", item["imageUrl"], e)

with open("catalog_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(catalog_embeddings, f, ensure_ascii=False)

print("catalog_embeddings.json created")