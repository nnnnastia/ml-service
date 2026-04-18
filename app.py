from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np
import json
import io

app = FastAPI()

IMG_SIZE = 224

CLASS_NAMES = ["hat", "pillow", "scarf", "sweater", "toy"]

UA_NAMES = {
    "hat": "Шапка",
    "pillow": "Подушка",
    "scarf": "Шарф",
    "sweater": "Світер",
    "toy": "Іграшка"
}

classifier_model = load_model("final_classifier.keras", compile=False, safe_mode=False)
embedding_model = load_model("final_embedding_model.keras", compile=False, safe_mode=False)

with open("catalog_embeddings.json", "r", encoding="utf-8") as f:
    catalog_embeddings = json.load(f)

def prepare_image(image: Image.Image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/search-by-photo")
async def search_by_photo(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Потрібно завантажити зображення")

    content = await file.read()

    try:
        image = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Некоректне зображення")

    x = prepare_image(image)

    # 1. Класифікація
    class_probs = classifier_model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(class_probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(class_probs[pred_idx])

    # 2. Embedding запиту
    query_embedding = embedding_model.predict(x, verbose=0)[0]

    # 3. Фільтр за категорією
    filtered_items = [
        item for item in catalog_embeddings
        if item["category"] == pred_class
    ]

    # 4. Similarity
    scored = []
    for item in filtered_items:
        sim = cosine_similarity(query_embedding, item["embedding"])
        scored.append({
            "productId": item["productId"],
            "similarity": float(sim)
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "predictedCategory": pred_class,
        "predictedCategoryUa": UA_NAMES.get(pred_class, pred_class),
        "confidence": confidence,
        "results": scored[:10]
    }