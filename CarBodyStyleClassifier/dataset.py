import os
import shutil
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from pathlib import Path


class ImageDataset:
    def __init__(self, data_dir, raw_data_dir):
        self.num_classes = 9
        self.data_dir = data_dir
        self.raw_data_dir = raw_data_dir
        self.train_ds = None
        self.test_ds = None

    def get_formatted_data(self):
        x_train, y_train = iter(self.train_ds).next()
        x_test, y_test = iter(self.test_ds).next()

        def rgb2gray(rgb):
            # np.expand_dims(np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]), axis=-1)
            return np.expand_dims(np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]), axis=-1)

        def format_data(x, y):
            x = np.array(x)
            x.astype('float32')
            
            # x = rgb2gray(x)
            x /= 255

            y = tf.keras.utils.to_categorical(y, self.num_classes)  # num classes

            return x, y

        x_train, y_train = format_data(x_train, y_train)
        x_test, y_test = format_data(x_test, y_test)

        return (x_train, y_train), (x_test, y_test)

    def create_dataset(self, img_size=(128, 128), batch_size=10000):
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.1,
            subset="training",
            seed=1020,
            shuffle=True,
            image_size=img_size,
            batch_size=batch_size)

        self.test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.1,
            subset="validation",
            seed=1020,
            shuffle=True,
            image_size=img_size,
            batch_size=batch_size)

    def delete_formatted_data(self):
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                os.remove(os.path.join(root, file))

    def format_raw(self, bulk=True):
        # Bulking the images allows a progress bar
        if bulk:
            images = []
            for f in os.scandir(self.raw_data_dir):
                images.append(f)

            for f in tqdm(images):
                # Gets path to class folder
                copy_dir = self.data_dir / Path(f.name.split('_')[-2])

                # Creates class folder if it does not exist
                if not os.path.exists(copy_dir):
                    os.makedirs(copy_dir)

                shutil.copy(f.path, copy_dir)
        else:
            for f in tqdm(os.scandir(self.raw_data_dir)):
                # Gets path to class folder
                copy_dir = self.data_dir/Path(f.name.split('_')[-2])

                # Creates class folder if it does not exist
                if not os.path.exists(copy_dir):
                    os.makedirs(copy_dir)

                shutil.copy(f.path, copy_dir)
