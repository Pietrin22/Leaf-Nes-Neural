#train
from data import load_plantvillage_dataset
from models import build_cnn, build_mlp
import argparse
import tensorflow as tf
import json
import os


def train_and_evaluate(model_name, color_mode, epochs, seed):
    if model_name == "mlp":
        img_size = (14,14)
    elif model_name == "cnn":
        img_size=(128,128)
    channels = 1 if color_mode == "grayscale" else 3

    train, val, test, class_names, num_classes, dataset = load_plantvillage_dataset(img_size=img_size, color_mode=color_mode)

    input_shape = img_size + (channels,)

    if model_name == "mlp":
        model = build_mlp(input_shape, num_classes)
    elif model_name == "cnn":
        model = build_cnn(input_shape, num_classes)
    else:
        raise ValueError("Modelo inválido")

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/{model_name}_{color_mode}_best.h5"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True)
    ]

    history = model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks)

    test_loss, test_acc = model.evaluate(test)
    print(f"Test acc: {test_acc:.4f}")

    os.makedirs("results", exist_ok=True)
    with open(f"results/history_{model_name}_{color_mode}.json", "w") as f:
        json.dump(history.history, f)

    return model, history, class_names
