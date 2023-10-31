import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import numpy as np


class DataLoaderService():
    data_path: str
    labels: list
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    train_images: list
    train_labels: list
    test_images: list
    test_labels: list
    valid_images: list
    valid_labels: list

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        pass

    def load_data(self):
        # Load CIFAR10 
        try:
            print(f"Loading data from path {self.data_path} \n")
            train_images = np.load(f'{self.data_path}/train_images.npy')
            train_labels = np.load(f'{self.data_path}/train_labels.npy')
            test_images = np.load(f'{self.data_path}/test_images.npy')
            test_labels = np.load(f'{self.data_path}/test_labels.npy')
        except FileNotFoundError:
            print("CIFAR-10 data not found. Downloading...")
            (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
            np.save(f'{self.data_path}/train_images.npy', train_images)
            np.save(f'{self.data_path}/train_labels.npy', train_labels)
            np.save(f'{self.data_path}/test_images.npy', test_images)
            np.save(f'{self.data_path}/test_labels.npy', test_labels)
            print("CIFAR-10 data downloaded and saved.")


        # Pre-process data
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')

        # Standardise images (255 is the total number of pixels an image can have)
        train_images = train_images / 255
        test_images = test_images / 255 

        # One Hot Encoding
        num_classes = len(self.class_names)
        train_labels = to_categorical(train_labels, num_classes)
        test_labels = to_categorical(test_labels, num_classes)

        # Get validation data
        train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

        # Set values to data
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.valid_images = valid_images
        self.valid_labels = valid_labels

        return (self.train_images, self.train_labels, self.valid_images, self.valid_labels)

    def get_training_set(self):
        return (self.train_images, self.train_labels)

    def get_testing_set(self):
        return (self.test_images, self.test_labels)
    
    def get_validation_set(self):
        return (self.valid_images, self.valid_labels)
    
    def get_classes(self):
        return self.class_names

    
    def visualise_data(self, num_images: int = 25):
        # Visualizing some of the images from the training dataset
        plt.figure(figsize=[10,10])
        for i in range (num_images):
            plt.subplot(5, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i], cmap=plt.cm.binary)
            plt.xlabel(self.class_names[self.train_labels[i][0]])

            plt.show()

