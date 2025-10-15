#data
import tensorflow as tf
import tensorflow_datasets as tfds

def load_plantvillage_dataset(img_size=(128,128), batch_size=32, color_mode="rgb"):
    """
    Carrega PlantVillage via tensorflow_datasets.
    Divide em treino/val/test com 70/15/15.
    """
    dataset, info = tfds.load("plant_village", split='train', with_info=True, as_supervised=True)
    num_classes = info.features['label'].num_classes
    class_names = info.features['label'].names
    dataset = dataset.take(6000)
    dataset_size = 6000
    def preprocess(image, label):
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        if color_mode == "grayscale":
            image = tf.image.rgb_to_grayscale(image)
        return image, label

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000, seed=42)

    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)

    train = dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val = dataset.skip(train_size).take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test = dataset.skip(train_size + val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train, val, test, class_names, num_classes, dataset
