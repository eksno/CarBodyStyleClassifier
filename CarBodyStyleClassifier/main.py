import os
import numpy as np
import tensorflow as tf

from pathlib import Path

from model import Model
from dataset import ImageDataset

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def main():
    if input('train? (y, N)'):
        (x_train, y_train), (x_test, y_test), samples = get_dataset(train=True)
        print('x_train shape:', x_train.shape)
        print('y_train shape:', y_train.shape)
        print('sample shape:', samples.shape)

        model = Model((x_train, y_train), (x_test, y_test), samples)

        model.create_model()

        model.model.summary()

        model.train_model()
    if 'y' == input('pred? (y, N)'):
        (x_train, y_train), (x_test, y_test), samples = get_dataset(train=False)
        print('sample shape:', samples.shape)

        model = Model((x_train, y_train), (x_test, y_test), samples)

        model.create_model()

        model.model.summary()

        model.run_model()


def get_dataset(train=True):
    src_dir = os.getcwd()

    if train:
        raw_data_dir = Path(str(src_dir + '\\raw_dataset'))
        data_dir = Path(str(src_dir + '\\dataset'))
    sample_dir = Path(str(src_dir + '\\test_images'))

    if train:
        dataset = ImageDataset(data_dir, raw_data_dir, sample_dir)
    else:
        dataset = ImageDataset(None, None, sample_dir)

    # if input('reformat data (y, N)') == 'y':
    #     dataset.delete_formatted_data()
    #     dataset.format_raw()


    dataset.create_dataset()
    return dataset.get_formatted_data()


if __name__ == '__main__':
    main()
