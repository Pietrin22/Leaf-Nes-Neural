import matplotlib.pyplot as plt
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_history(history_file):
    with open(history_file, "r") as f:
        hist = json.load(f)
    plt.plot(hist["accuracy"], label="train acc")
    plt.plot(hist["val_accuracy"], label="val acc")
    plt.legend()
    plt.show()


def plot_confusion_matrix(model, dataset, class_names):
    y_true, y_pred = [], []
    for x, y in dataset:
        preds = model.predict(x)
        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.show()