import tensorflow as tf
import numpy as np
import cv2
from keras_vggface.vggface import VGGFace
import os
import pandas as pd

from data_extraction import get_filenames, get_dataset

class ImageClassifier:
    def __init__(self):
        self.labels = None
        self.model = None

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

        # data augmentation
        # create 2 new samples for each image and add them to the dataset
        new_samples = []
        new_labels = []
        for i in range(len(images)):
            for j in range(5):
                # rotate the image
                rows, cols, _ = images[i].shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), np.random.randint(-10, 10), 1)
                augmented_image = cv2.warpAffine(images[i], M, (cols, rows))

                # random left-right flip
                if np.random.randint(0, 2) == 0:
                    augmented_image = cv2.flip(augmented_image, 1)

                # random brightness and contrast
                alpha = np.random.uniform(0.5, 1.5)
                beta = np.random.uniform(-10, 10)
                augmented_image = cv2.convertScaleAbs(augmented_image, alpha=alpha, beta=beta)

                # add the new sample to the dataset
                new_samples.append(augmented_image)

                # get the index of the label in the self.labels array
                index = np.where(self.labels == filenames[i])[0][0]

                # add the new sample to the labels array
                new_labels.append(labels[index])

        # convert the new samples to numpy array
        X = np.array(new_samples)
        Y = np.array(new_labels)

        # Preprocess the images
        X = self.preprocess_images(X)

        initializer = tf.keras.initializers.he_normal(seed=32)

        input_shape_densenet = (224, 224, 3)

        densenet_model = tf.keras.applications.DenseNet121(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=input_shape_densenet,
            pooling=None
        )

        densenet_model.trainable = True

        for layer in densenet_model.layers:
            if 'conv5' in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False

        self.model = tf.keras.Sequential([
            # resize images to 224x224
            tf.keras.layers.experimental.preprocessing.Resizing(224, 224, interpolation='bilinear', input_shape=(64, 64, 3)),
            densenet_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=512,
                                activation='relu',
                                kernel_initializer=initializer),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=256,
                            activation='relu',
                            kernel_initializer=initializer),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(units=len(self.labels),
                            activation='softmax',
                            kernel_initializer=initializer)
        ])

        val_dataset, val_labels = get_dataset('data/Ephemere/labels/')

        # Extract the features
        val_x = self.preprocess_images(val_dataset)

        val_y = np.zeros((len(val_labels), len(self.labels)), dtype=np.float32)
        for i in range(len(val_labels)):
            index = np.where(self.labels == val_labels[i])[0]
            val_y[i][index] = 1

        # Compile the model
        self.model.compile(loss='binary_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['accuracy'])
        
        self.model.fit(X, Y, epochs=10, batch_size=32, validation_data=(val_x, val_y))

        # Save the model
        self.model.save('model_image')

        # Save the labels in a csv file
        pd.DataFrame(self.labels).to_csv('labels_image.csv', index=False, header=False)

    # Load the model
    def load_model(self):
        self.model = tf.keras.models.load_model('model_image')

        # Read the labels from the csv file and convert them to a string array and flatten it
        self.labels = pd.read_csv('labels_image.csv', header=None).values.flatten()

    # Preprocess the image
    def preprocess_images(self, images):
        # Resize the image
        images = np.array([cv2.resize(img, (64, 64)) for img in images])

        # Convert the image to float32
        images = images.astype(np.float32)

        # Normalize the image
        images /= 255

        return images

    # Classify the image
    def classify(self, images):
        # Preprocess the images
        images = self.preprocess_images(images)

        # Predict the class
        prediction = self.model.predict(images, verbose=0)

        # Get the class with the highest probability for each image
        class_index = np.argmax(prediction, axis=1)

        # Get the class name
        class_name = [self.labels[i] for i in class_index]

        # Get the confidence
        confidence = [prediction[i][j] for i, j in enumerate(class_index)]
        
        # Return the class name and the probability
        return class_name, confidence