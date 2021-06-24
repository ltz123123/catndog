import os
from tqdm import tqdm
import numpy as np
from shutil import copyfile


categories = ["Cat", "Dog"]
img_parent_dir = r"C:\Users\juste\PycharmProjects\catndog\kagglecatsanddogs_3367a\PetImages"
train_test_sub_dir = ["train", "test"]

test_data_idx = np.random.choice(
    np.arange(12500),
    size=12500 // 5,
    replace=False
)


def create_directory():
    for category in categories:
        for sub_dir in train_test_sub_dir:
            os.makedirs(
                os.path.join("image_data", sub_dir, category),
                exist_ok=True
            )


def train_test_split():
    for category in categories:
        img_dir = os.path.join(img_parent_dir, category)
        for img_file in tqdm(os.listdir(img_dir)):
            num = int(img_file.split(".")[0])
            copyfile(
                os.path.join(img_dir, img_file),
                os.path.join(
                    "image_data",
                    "test" if num in test_data_idx else "train",
                    category,
                    img_file
                )
            )


if __name__ == "__main__":
    create_directory()
    train_test_split()





