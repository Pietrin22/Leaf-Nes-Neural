import tensorflow as tf


def load_leaf_dataset(img_size=(128, 128), batch_size=32, color_mode="rgb"):
    """
    Carrega o dataset de Leaf Classification a partir do Keras.
    Divide em treino/val/test com 80/10/10.
    """
    (train_ds, val_ds, test_ds), ds_info = tf.keras.utils.get_file(
        "leaf_classification",
        origin="https://storage.googleapis.com/download.tensorflow.org/example_images/leaf_classification.tar.gz",
        untar=True
    ), None

    # tf.keras.utils.get_file retorna o diretório do dataset
    data_dir = train_ds

    train = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        color_mode=color_mode
    )
    val = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        color_mode=color_mode
    )

    # Separar parte do val como teste
    val_batches = tf.data.experimental.cardinality(val)
    test = val.take(val_batches // 2)
    val = val.skip(val_batches // 2)

    AUTOTUNE = tf.data.AUTOTUNE
    train = train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val = val.cache().prefetch(buffer_size=AUTOTUNE)
    test = test.cache().prefetch(buffer_size=AUTOTUNE)

    class_names = train.class_names
    num_classes = len(class_names)

    return train, val, test, class_names, num_classes