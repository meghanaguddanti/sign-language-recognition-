import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from PIL import Image
import tensorflow as tf
from config import MODEL_PATH, LABELS_PATH, IMG_SIZE

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No model at {MODEL_PATH}\nRun: cd src   then: python train.py")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        class_names = json.load(f)
    print(f"Model loaded. Classes: {class_names}")
    return model, class_names

def preprocess(img):
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(img, model, class_names, top_k=5):
    probs = model.predict(preprocess(img), verbose=0)[0]
    top_k = min(top_k, len(class_names))
    top   = probs.argsort()[-top_k:][::-1]
    return {class_names[i]: float(probs[i]) for i in top}