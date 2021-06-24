import numpy as np
import tensorflow as tf
from time import strftime, localtime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


def create_model():
    model = Sequential()

    model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(256, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(256, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"]
    )

    return model


def train_model():
    data_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_data_gen = data_gen.flow_from_directory(
        directory="image_data/train/",
        class_mode="binary",
        batch_size=32,
        target_size=(224, 224),
    )
    val_data_gen = data_gen.flow_from_directory(
        directory="image_data/test",
        class_mode="binary",
        batch_size=32,
        target_size=(224, 224),
    )

    model = create_model()
    print(model.summary())

    time_now = strftime("%Y_%m_%d_%H_%M", localtime())
    NAME = f"cat_dog_model_{time_now}"
    tensorboard = TensorBoard(log_dir=r"logs\{}".format(NAME))
    checkpoint = ModelCheckpoint(
        r"saved_models\{}".format("cat_dog_{epoch:02d}_{val_loss:.3f}"),
        monitor='val_loss',
        mode='min'
    )

    model.fit(
        train_data_gen,
        validation_data=val_data_gen,
        epochs=1,
        callbacks=[tensorboard, checkpoint]
    )


if __name__ == "__main__":
    train_model()
