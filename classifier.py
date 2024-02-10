"""
TO DO
more preprocessing (ImageDataGenerator?)
tensorboard
consider how the model would be deployed (Docker)
"""
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import datetime

from keras.utils import normalize
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout

# config
# desired size of images (IMG_SIZE x IMG_SIZE)
IMG_SIZE = 64
# x axis co-ordinates of results to be displayed after training (for clarity)
RESULTS_WINDOW_LOWER, RESULTS_WINDOW_UPPER = 200,300


class ImageIngestion:
    """
    read in and pre-process images (following extract / transform / load pipeline)

    Args:
        None
    
    Attributes:
        None
    """
    def __init__(self):
        pass

    def image_pre_process(self, path):
        '''
        function to read in images and labels
        removes all images containing both military and civilian vehicles
        appends labels to a list
        reads in images and isolates those pixels within the bounding box, then resizes
        assigns the resized images to a numpy array and returns the normalized data

        Args:
            path (string): path to images
            image_size (int): size of image after resizing
            samples (int): number of images to load (all 6772 caused memory issues on this laptop)

        Returns:
            normalised array of image pixel values and array of label values
        '''
        
        labels_df = pd.read_csv(path)

        files = labels_df.filename

        # identifies files with greater than 1 bounding box
        values, counts = np.unique(files, return_counts=True)

        df_unique = pd.DataFrame({'values': values,'counts':counts})
        df_unique.drop(df_unique[df_unique['counts'] == 1].index, inplace = True)

        # removes those filenames with both military and civilian vehicles
        both_list =[]
        matchers = ['military', 'civilian']
        for filename in df_unique.values:
            vehicle_types = labels_df.loc[labels_df['filename'] == filename[0], 'class']
            if all(x in ' '.join(vehicle_types) for x in matchers):
                both_list.append(filename[0])

        labels_df = labels_df[~labels_df['filename'].isin(both_list)]

        # extracts bounding box and assigns resized image to imgs list
        imgs = []
        non_existent = []
        for i in range(len(labels_df)):
            try:
                img = cv2.imread('Images//{}'.format(labels_df.filename.iloc[i]))[labels_df.ymin.iloc[i]:labels_df.ymax.iloc[i], labels_df.xmin.iloc[i]:labels_df.xmax.iloc[i]]
                img_pil = Image.fromarray(img, 'RGB')
                resized_pil = img_pil.resize((IMG_SIZE, IMG_SIZE))
                imgs.append(np.array(resized_pil))
            except TypeError:
                print('Sample {} does not exist'.format(i))
                non_existent.append(i)
                continue

        labels_df = labels_df.drop(labels_df.index[non_existent])
        
        # converts to a binary classification problem (military = 1, civilian = 0)
        labels = []
        [labels.append(1) if 'military' in labels_df['class'].iloc[i] else labels.append(0) for i in range(len(labels_df))]

        imgs_arr = np.array(imgs)
        labels_arr = np.array(labels)

        return normalize(imgs_arr, axis=1), labels_arr


class BinaryClassifier:
    """
    create, compile, run and save binary classifier

    Args:
        train_imgs (array): np array of pre-processed training images
        train_labels (array): np array of training labels (0s and 1s)
        test_imgs (array): np array of pre-processed testing images
        test_labels (array): np array of testing labels (0s and 1s)

    
    Attributes:
        model (keras.Sequential): Keras Sequential model for classification
        model_version (maybe str / float / datetime): some unique identifier for model iteration
        train_imgs (array): np array of pre-processed training images
        train_labels (array): np array of training labels (0s and 1s)
        test_imgs (array): np array of pre-processed testing images
        test_labels (array): np array of testing labels (0s and 1s)
        history (History object): history of model training statistics
    """
    def __init__(self, train_imgs, train_labels, test_imgs, test_labels):
        # ensure results are reproducible
        np.random.seed(0)
        self.model = Sequential()
        self.model_version = None
        self.train_imgs = train_imgs
        self.train_labels = train_labels
        self.test_imgs = test_imgs
        self.test_labels = test_labels
        self.history = None

    def documentation(self):
        """
        code to save model / hyperparameters / performance / outputs etc for development iteration
        unique model directory for each model iteration
        something with self.model_version
        """
        self.model_version = datetime.datetime.now () # for example
        pass


    def create_and_compile_model(self, num_layers: int, num_filters: int, filter_shape: tuple, pool_size: tuple, dropout: float):
        """
        build and compile model

        Args:
            num_layers (int): number of hidden layers
            num_filters (int): number of filters in Conv2D layers
            filter_shape (tuple): shape of filter in Conv2D layers (n x n)
            pool_size (type): size of pool for pooling layers (m x m)

        Returns:
            compiled model
        """
        input_shape = (IMG_SIZE, IMG_SIZE, 3)

        self.model.add(Conv2D(num_filters, filter_shape, input_shape=input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=pool_size))
        self.model.add(Dropout(dropout))    

        for _ in range(num_layers):
            self.model.add(Conv2D(num_filters, filter_shape, kernel_initializer = 'he_uniform'))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=pool_size))
            self.model.add(Dropout(dropout))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))

        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid')) 

        self.model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate = 0.0001),
                  metrics=['accuracy'])

    def run_model(self, epochs):
        """
        run model for specified number of epochs
        expand to save trained model to model directory

        Args:
            epochs (int): number of epochs

        Returns:
            fitted model
        """
        self.history = self.model.fit(self.train_imgs, 
                             self.train_labels, 
                             batch_size = 1, 
                             verbose = 1, 
                             epochs = epochs,      
                             shuffle = True,
                             validation_data=(self.test_imgs, self.test_labels)
                         )

class MakePredictions:
    """
    takes trained model from BinaryClassifier class and makes predictions on image set

    Args:
        history (History object): history of model training statistics
        imgs (array): np array of images upon when to make predictions (can be testing or training)
        labels (array): np array of labels related to the imgs array
        model (keras.Sequential): keras Sequential model trained by the run_model method of the BinaryClassifier class

    Attributes:
        history (History object): history of model training statistics
        imgs (array): np array of images upon when to make predictions (can be testing or training)
        labels (array): np array of labels related to the imgs array
        model (keras.Sequential): keras Sequential model trained by the run_model method of the BinaryClassifier class
        results (list): empty list in which to append test vs predicted data
    """
    def __init__(self, history, imgs, labels, model):
        self.history = history
        self.imgs = imgs
        self.labels = labels
        self.model = model
        self.results = []

    def make_predictions(self):
        """
        generates predictions
        expand to save results to model directory

        Args:
            None

        Returns:
            list of list of test data vs prediction - [[],[],[]] etc
        """
        for n in range(len(self.imgs)):
            single_image = self.imgs[n]
            img = np.expand_dims(single_image,axis=0)
            self.results.append([self.labels[n], self.model.predict(img)[0][0]])

    def plot_predictions(self):
        """
        visualises binary predictions
        expand to save accuracy etc to model directory

        Args:
            None

        Retuns:
            plot of test vs predicted (0s and 1s vs prediction)
            plot of training accuracy and validation accuracy
        """
        plt.figure(1)
        plt.plot(self.results[int(RESULTS_WINDOW_LOWER): int(RESULTS_WINDOW_UPPER)])
     
        plt.figure(2)
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label='validation accuracy')
        plt.legend()

        plt.show()
