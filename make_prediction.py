import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
# from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


model = tf.keras.models.load_model("saved_models/cat_dog_01_0.624")
data_dir = r"C:\Users\juste\PycharmProjects\catndog\test_img"

for img_name in os.listdir(data_dir):
    try:
        img = img_to_array(
            load_img(
                os.path.join(data_dir, img_name),
                target_size=(224, 224),
            ),
            dtype=int
        )

        prediction = model.predict(
            img.reshape((-1, 224, 224, 3))
        )
        print(prediction)
        plt.imshow(img)
        plt.title(f"{prediction[0][0]}")
        plt.show()

    except:
        continue
