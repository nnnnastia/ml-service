from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np
import json
import io
import os
import traceback

IMG_SIZE = 224

CLASS_NAMES = ["hat", "pillow", "scarf", "sweater", "toy"]

UA_NAMES = {
    "hat": "Шапка",
    "pillow": "Подушка",
    "scarf": "Шарф",
    "sweater": "Світер",
    "toy": "Іграшка"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSIFIER_PATH = os.path.join(BASE_DIR, "final_classifier.keras")
EMBEDDING_PATH = os.path.join(BASE_DIR, "final_embedding_model.keras")
EMBEDDINGS_JSON_PATH = os.path.join(BASE_DIR, "catalog_embeddings.json")

classifier_model = None
embedding_model = None
catalog_embeddings = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier_model, embedding_model, catalog_embeddings

    try:
        print("BASE_DIR:", BASE_DIR)
        print("FILES IN DIR:", os.listdir(BASE_DIR))

        print("Classifier exists:", os.path.exists(CLASSIFIER_PATH), CLASSIFIER_PATH)
        print("Embedding exists:", os.path.exists(EMBEDDING_PATH), EMBEDDING_PATH)
        print("Embeddings json exists:", os.path.exists(EMBEDDINGS_JSON_PATH), EMBEDDINGS_JSON_PATH)

        print("Loading classifier model...")
        classifier_model = load_model(CLASSIFIER_PATH, compile=False, safe_mode=False)
        print("Classifier loaded")

        print("Loading embedding model...")
        embedding_model = load_model(EMBEDDING_PATH, compile=False, safe_mode=False)
        print("Embedding model loaded")

        print("Loading catalog embeddings json...")
        with open(EMBEDDINGS_JSON_PATH, "r", encoding="utf-8") as f:
            catalog_embeddings = json.load(f)
        print("Catalog embeddings loaded")

        print("ML models loaded successfully")

    except Exception as e:
        print("STARTUP ERROR:", str(e))
        traceback.print_exc()
        raise e

    yield


app = FastAPI(lifespan=lifespan)


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


@app.get("/")
def root():
    return {"message": "ML service is running"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/search-by-photo")
async def search_by_photo(file: UploadFile = File(...)):
    global classifier_model, embedding_model, catalog_embeddings

    if classifier_model is None or embedding_model is None or catalog_embeddings is None:
        raise HTTPException(status_code=503, detail="ML models are not loaded yet")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Потрібно завантажити зображення")

    content = await file.read()

    try:
        image = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Некоректне зображення")

    x = prepare_image(image)

    class_probs = classifier_model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(class_probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(class_probs[pred_idx])

    query_embedding = embedding_model.predict(x, verbose=0)[0]

    filtered_items = [
        item for item in catalog_embeddings
        if item["category"] == pred_class
    ]

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