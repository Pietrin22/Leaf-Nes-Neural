import argparse
import tensorflow as tf
import json
import os

from data import load_leaf_dataset
from models import build_mlp, build_cnn


def train_and_evaluate(model_name, color_mode, epochs, seed):
    img_size = (128, 128)
    channels = 1 if color_mode == "grayscale" else 3

    train, val, test, class_names, num_classes = load_leaf_dataset(img_size=img_size, color_mode=color_mode)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["mlp", "cnn"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--color_mode", type=str, choices=["rgb", "grayscale"], default="rgb")
    args = parser.parse_args()

    tf.random.set_seed(args.seed)

    train_and_evaluate(args.model, args.color_mode, args.epochs, args.seed)