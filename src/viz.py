#viz
from data import load_plantvillage_dataset
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_history(hist, title):
    plt.title(title)
    plt.plot(hist.history["accuracy"], label="train acc")
    plt.plot(hist.history["val_accuracy"], label="val acc")
    plt.plot(hist.history["loss"], label="train loss")
    plt.plot(hist.history["val_loss"], label="val loss")
    plt.legend()
    plt.show()

def plot_confusion_matrix(model_name, model, class_names, color_mode="rgb"):
    """
    Gera matriz de confusão automaticamente, detectando formato de entrada do modelo.
    Funciona com CNNs e MLPs, independente do tamanho da imagem.
    """
    if model_name == "mlp":
        img_size = (14,14)
    elif model_name == "cnn":
        img_size=(128,128)

    train, val, test, class_names, num_classes, dataset = load_plantvillage_dataset(img_size=img_size, color_mode=color_mode)

    y_true, y_pred = [], []
    for x, y in test:
        preds = model.predict(x, verbose=0)
        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    # Matriz de confusão normalizada
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, cmap="Blues")

    plt.title(f"Matriz de Confusão (Normalizada) - {model_name.upper()}")
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Verdadeira")
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=90, fontsize=8)
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names, fontsize=8)
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()
