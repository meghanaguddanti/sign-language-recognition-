import sys
import os

# Fix paths
APP_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(APP_DIR)
SRC_DIR     = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

# Override config paths to use absolute paths
os.environ["SLR_ROOT"] = ROOT_DIR

import gradio as gr

# ── Load model ─────────────────────────────────────────────────────────────────
import json
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH  = os.path.join(ROOT_DIR, "models", "best_model.keras")
LABELS_PATH = os.path.join(ROOT_DIR, "models", "class_names.json")

print(f"Loading model from: {MODEL_PATH}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")
print(f"Labels exist: {os.path.exists(LABELS_PATH)}")

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH) as f:
    class_names = json.load(f)

print(f"Model loaded! Classes: {class_names}")

IMG_SIZE = (64, 64)

def preprocess(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def run_prediction(image):
    if image is None:
        return {"No image provided": 1.0}
    probs    = model.predict(preprocess(image), verbose=0)[0]
    top      = probs.argsort()[-5:][::-1]
    return {class_names[i]: float(probs[i]) for i in top}

# ── UI ─────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Sign Language Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✋ Sign Language Recognition\nIdentify ASL hand signs A–Z.")

    with gr.Tabs():
        with gr.Tab("📁 Upload Image"):
            with gr.Row():
                img_in  = gr.Image(type="pil", label="Upload hand sign", height=300)
                img_out = gr.Label(num_top_classes=5, label="Predictions")
            gr.Button("Identify Sign", variant="primary").click(
                run_prediction, inputs=img_in, outputs=img_out)

        with gr.Tab("📷 Webcam"):
            with gr.Row():
                cam_in  = gr.Image(sources=["webcam"], type="pil",
                                   streaming=False, label="Webcam", height=300)
                cam_out = gr.Label(num_top_classes=5, label="Predictions")
            gr.Button("Capture & Identify", variant="primary").click(
                run_prediction, inputs=cam_in, outputs=cam_out)

    gr.Markdown("_MobileNetV2 · TensorFlow · Gradio_")

if __name__ == "__main__":
    print("Starting Gradio app...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)