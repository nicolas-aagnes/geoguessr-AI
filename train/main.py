import os

import tensorflow as tf
from model import get_model
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    dataset_params = {
        "directory": "data",
        "image_size": (640, 640),
        "validation_split": 0.2,
        "seed": 123,
    }

    dataset_train = tf.keras.preprocessing.image_dataset_from_directory(subset="training", **dataset_params)

    dataset_val = tf.keras.preprocessing.image_dataset_from_directory(subset="validation", **dataset_params)

    model = get_model()
    print(model.summary())

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/checkpoint",
        monitor="val_sparse_categorical_accuracy",
        save_weights_only=True,
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    model.fit(
        x=dataset_train,
        validation_data=dataset_val,
        callbacks=[checkpoint, keras.callbacks.ReduceLROnPlateau()],
        epochs=1,
    )


if __name__ == "__main__":
    main()
