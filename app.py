from data import load_plantvillage_dataset
from models import build_cnn, build_mlp
from viz import plot_history, plot_confusion_matrix
import streamlit as st
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from sklearn.metrics import confusion_matrix

st.title("🌿 Classificação de Folhas - PlantVillage")
st.sidebar.header("Configurações do Modelo")

model_name = st.sidebar.selectbox("Modelo", ["mlp", "cnn"])
color_mode = st.sidebar.selectbox("Modo de Cor", ["rgb", "grayscale"])
epochs = st.sidebar.slider("Épocas", 1, 50, 5)
batch_size = st.sidebar.slider("Batch Size", 8, 64, 16)
train_button = st.sidebar.button("Treinar Modelo")

if train_button:
    with st.spinner("Carregando dados e treinando modelo..."):
        img_size = (14,14) if model_name == "mlp" else (128,128)
        channels = 1 if color_mode == "grayscale" else 3
        input_shape = img_size + (channels,)

        train, val, test, class_names, num_classes = load_plantvillage_dataset(img_size=img_size, batch_size=batch_size, color_mode=color_mode)

        model = build_mlp(input_shape, num_classes) if model_name == "mlp" else build_cnn(input_shape, num_classes)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train, validation_data=val, epochs=epochs, verbose=1)

    st.success("✅ Treinamento concluído!")
    plot_history(history.history)

    st.write("### Avaliação no conjunto de teste")
    test_loss, test_acc = model.evaluate(test)
    st.metric("Acurácia no Teste", f"{test_acc:.4f}")
    st.metric("Loss no Teste", f"{test_loss:.4f}")

    st.write("### Matriz de Confusão")
    plot_confusion_matrix(model, test, class_names)

    if st.button("Salvar Modelo"):
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/{model_name}_{color_mode}.h5"
        model.save(path)
        st.success(f"Modelo salvo em: {path}")