import tensorflow as tf
import numpy as np
import cv2
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import os
import pandas as pd

from data_extraction import get_filenames, get_dataset

class Classifier:
    def __init__(self):
        self.labels = None
        self.model = None

        # Create the VGGFace model
        self.vgg_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

        # Create a facenet model
        # self.facenet = tf.keras.models.load_model('facenet_keras.h5')

    # Create and train the model
    def train_model(self, folders):
        # Read the images in the folders
        filepaths = []
        for folder in folders:
            filepaths += get_filenames(folder)

        # Create a label array from the name of the image
        filenames = [os.path.splitext(os.path.basename(f))[0] for f in filepaths]

        # Create the labels
        self.labels = np.unique(filenames)

        # Read the images in the folder
        images = np.array([cv2.imread(f) for f in filepaths])

        # Create the labels 
        labels = np.zeros((len(self.labels), len(self.labels)), dtype=np.float32)
        for i in range(len(self.labels)):
            labels[i][i] = 1

        # Resize the images to 64x64
        images = np.array([cv2.resize(image, (512, 512)) for image in images])

        # data augmentation
        # create 5 new samples for each image and add them to the dataset
        new_samples = []
        new_labels = []
        for i in range(len(images)):
            for j in range(5):
                # rotate the image
                rows, cols, _ = images[i].shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), np.random.randint(-15, 15), 1)
                augmented_image = cv2.warpAffine(images[i], M, (cols, rows))

                # random left-right flip
                if np.random.randint(0, 2) == 0:
                    augmented_image = cv2.flip(augmented_image, 1)

                # random brightness and contrast
                alpha = np.random.uniform(0.5, 1.5)
                beta = np.random.uniform(-10, 10)
                augmented_image = cv2.convertScaleAbs(augmented_image, alpha=alpha, beta=beta)

                # # random noise
                # noise = np.random.randint(-10, 10, augmented_image.shape)
                # augmented_image = augmented_image + noise
                # augmented_image = np.clip(augmented_image, 0, 255)

                # add the new sample to the dataset
                new_samples.append(augmented_image)

                # get the index of the label in the self.labels array
                index = np.where(self.labels == filenames[i])[0][0]

                # add the new sample to the labels array
                new_labels.append(labels[index])

        # convert the new samples to numpy array
        dataset = np.array(new_samples)
        y = np.array(new_labels)

        # Extract the features
        x = self.extract_features(dataset)
        
        # Shuffle the dataset
        p = np.random.permutation(len(x))
        x = x[p]
        y = y[p]

        dropout = 0.5
        # Create the model with dropout and data normalization
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1024, activation='relu', input_shape=(x.shape[1],)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(len(self.labels), activation='softmax')
        ])

        val_dataset, val_labels = get_dataset('data/Ephemere/labels/')

        # Extract the features
        val_x = self.extract_features(val_dataset)

        val_y = np.zeros((len(val_labels), len(self.labels)), dtype=np.float32)
        for i in range(len(val_labels)):
            index = np.where(self.labels == val_labels[i])[0]
            val_y[i][index] = 1

        # Compile the model
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Train the model
        self.model.fit(x, y, epochs=15, batch_size=32, validation_data=(val_x, val_y))

        # Save the model
        self.model.save('model')

        # Save the labels in a csv file
        pd.DataFrame(self.labels).to_csv('labels.csv', index=False, header=False)

    # Load the model
    def load_model(self):
        self.model = tf.keras.models.load_model('model')

        # Read the labels from the csv file and convert them to a string array and flatten it
        self.labels = pd.read_csv('labels.csv', header=None).values.flatten()

    # Extract the features from the images
    def extract_features(self, images):
        # Resize the images
        images = np.array([cv2.resize(img, (224, 224)) for img in images])

        # # Convert the images to float32
        images = images.astype(np.float32)

        # Normalize the image
        images[..., 0] -= 91.4953
        images[..., 1] -= 103.8827
        images[..., 2] -= 131.0912

        # Extract the features
        #images = utils.preprocess_input(images, version=2)
        features = self.vgg_features.predict(images, verbose = 0)

        return features

    # Classify the images
    def classify(self, images):
        # Preprocess the images
        features = self.extract_features(images)

        # Predict the class
        prediction = self.model.predict(features, verbose=0)

        # Get the class with the highest probability for each image
        class_index = np.argmax(prediction, axis=1)

        # Get the class name
        class_name = [self.labels[i] for i in class_index]

        # Get the confidence
        confidence = [prediction[i][j] for i, j in enumerate(class_index)]
        
        # Return the class name and the probability
        return class_name, confidence