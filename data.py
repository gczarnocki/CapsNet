import numpy as np
import gzip
import tensorflow as tf
from keras.datasets import fashion_mnist, mnist


class Data(object):
    def __init__(self, train_images, train_labels, test_images, test_labels):
        self._train = DataSet(train_images, train_labels)
        self._test = DataSet(test_images, test_labels)

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test
    
    @property
    def validation(self):
        return self._test


class DataSet(object):
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]


    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate(
                (images_rest_part, images_new_part), axis=0), np.concatenate(
                    (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def load_data(dataset_type, extended_dataset):
    if dataset_type in ('mnist', 'fashion_mnist'):
        if extended_dataset:
            return load_extended_dataset(dataset_type)
        else:
            return load_dataset(dataset_type)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset_type)


def get_dataset_from_type(dataset_type):
    if dataset_type == "mnist":
        return mnist.load_data()
    elif dataset_type == "fashion_mnist":
        return fashion_mnist.load_data()
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset_type)


def reshape_normalize_dataset(dataset, array_size):
    dataset = dataset.reshape(-1, array_size)
    dataset = np.multiply(dataset, 1.0 / 255.0)

    return dataset


def load_dataset(dataset_type):
    dataset = get_dataset_from_type(dataset_type)

    # train data -> data -> first element
    img_size = dataset[0][0][0].size

    train_images, train_labels = dataset[0]
    test_images, test_labels = dataset[1]

    train_images = reshape_normalize_dataset(train_images, img_size)
    test_images = reshape_normalize_dataset(test_images, img_size)

    return Data(train_images, train_labels, test_images, test_labels)


def rotate_array(array, rotations_no):
    return np.rot90(array, rotations_no)


def load_extended_dataset(dataset_type):
    dataset = get_dataset_from_type(dataset_type)

    # train data -> data -> first element
    img_size = dataset[0][0][0].size

    train_images, train_labels = dataset[0]
    test_images, test_labels = dataset[1]

    new_shape = list()
    new_shape.append(train_images.shape[0] * 4)
    new_shape = new_shape + list(train_images.shape[1:])

    new_train_images = np.zeros(new_shape)

    for i in range(0, train_images.shape[0]):
        for j in range(0, 4):
            new_train_images[4 * i + j] = rotate_array(train_images[i], j)

    train_images = reshape_normalize_dataset(train_images, img_size)
    test_images = reshape_normalize_dataset(test_images, img_size)

    return Data(train_images, train_labels, test_images, test_labels)
