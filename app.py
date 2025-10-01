import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("Plant Village Classification App")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--color-mode", type=str, choices=["rgb", "grayscale"], default="rgb")
args, _ = parser.parse_known_args()

model = tf.keras.models.load_model(args.model_path)
color_mode = args.color_mode

uploaded = st.file_uploader("Upload uma imagem de planta", type=["jpg", "png"])
if uploaded:
    img = Image.open(uploaded).resize((128,128))
    if color_mode == "grayscale":
        img = img.convert("L")
        arr = np.expand_dims(np.array(img), -1)
    else:
        arr = np.array(img)
    arr = np.expand_dims(arr/255.0, 0)
    preds = model.predict(arr)[0]
    pred_class = np.argmax(preds)
    st.write(f"Classe prevista: {pred_class}")
    st.bar_chart(preds)