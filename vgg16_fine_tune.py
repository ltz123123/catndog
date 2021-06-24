import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def build_model():
    model = VGG16(
        include_top=False,
        input_shape=(224, 224, 3)
    )
    model.trainable = False

    x = Flatten()(model.layers[-1].output)
    x = Dense(256, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=model.inputs, outputs=x)
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(lr=0.001),
                  metrics=["accuracy"])

    return model


def train_model():
    model = build_model()

    data_generator = ImageDataGenerator(rescale=1 / 255)
    train_data_gen = data_generator.flow_from_directory(
        "image_data/train/", batch_size=32, target_size=(224, 224), class_mode="binary"
    )
    test_data_gen = data_generator.flow_from_directory(
        "image_data/test/", batch_size=32, target_size=(224, 224), class_mode="binary"
    )

    callbacks = [
        ModelCheckpoint(r"saved_model\ft_vgg_model_at_{epoch}", monitor='val_loss', mode='min'),
        TensorBoard(log_dir=r"logs\catndog_model")
    ]

    history = model.fit(
        train_data_gen,
        validation_data=test_data_gen,
        epochs=50,
        verbose=2,
        # callbacks=callbacks
    )


if __name__ == "__main__":
    train_model()







