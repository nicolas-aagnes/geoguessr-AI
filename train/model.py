import tensorflow as tf
from tensorflow import keras


def get_model(input_shape=(640, 640, 3), num_classes=109):
    base_model = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")(x)
    x = tf.keras.layers.experimental.preprocessing.RandomZoom((-0.15, 0.0), None, "reflect")(x)

    x = base_model(x, training=False)

    x = keras.layers.Conv2D(512, 4, 2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(512, 4, 2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(num_classes, 3, 1)(x)
    outputs = keras.layers.Flatten()(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    return model
