import numpy as np
import cv2
import os
from keras_vggface.vggface import VGGFace

# Class features index
# Ordered list of the mean of features of all images of a person
# Each element of the list is a number

class Index:
    def __init__(self):
        self.index = []

    # Add the tuple (mean and name) at the right place in the list
    def add(self, features, name):
        # get the mean of the tuple to add
        mean = np.linalg.norm(features)

        # if the name is already in the list, update the features mean and the position
        if name in self.index:
            index = self.index.index(name)

            # calc the the new mean
            new_mean = (self.index[index][0] + mean) / 2

            # remove the old tuple
            self.index.pop(index)

            # add the new tuple at the right place
            self.add(new_mean, name)

        # if the name is not in the list, add the tuple at the right place
        else:
            # if the list is empty, add the tuple at the beginning
            if len(self.index) == 0:
                self.index.append((mean, name))

            # if the list is not empty, add the tuple at the right place
            else:
                # get the position where to add the tuple
                index = self.get_index(mean)
                # add the tuple at the right place
                self.index.insert(index, (mean, name))

    # Get the index where to add the tuple
    def get_index(self, mean):
        index = 0
        for i in range(len(self.index)):
            if mean < self.index[i][0]:
                index = i
                break
            else:
                index = i + 1
        return index
    
    # Get the name of the person
    def get_name(self, features):
        # if the list is empty, return None
        if len(self.index) == 0:
            return None
        
        mean = np.linalg.norm(features)
        index = self.get_index(mean)

        # check if the index is not the out of the list
        if index >= len(self.index):
            index = len(self.index) - 1
        
        else:
            # check if the the index is close to the previous index
            if index > 0 and abs(mean - self.index[index - 1][0]) < abs(mean - self.index[index][0]):
                index -= 1

        return self.index[index][1]

    # Pettry print the list
    def pretty_print(self):
        print("Index:")
        for i in range(len(self.index)):
            print("    ", self.index[i])

# Class indexer of Images
# Create an index from all images in a folder and subfolders
# Find the name of a person from an image
class Indexer:
    def __init__(self, folder):
        self.folder = folder

        # Read the images in the folder
        self.images_filenames = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

        # Create a label array from the name of the image
        self.labels = [f.split('.')[0] for f in self.images_filenames]

        # Create the VGGFace model
        self.vgg_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

        # Create the index
        self.index = Index()

        # for each image, get the features and add them to the index
        for i in range(len(self.images_filenames)):
            # read the image
            image = cv2.imread(os.path.join(folder, self.images_filenames[i]))

            # get the features
            features = self.extract_features(image)

            # add the features to the index
            self.index.add(features, self.labels[i])

    # Get the name of the person from an image
    def get_name(self, image):
        # get the features
        features = self.extract_features(image)

        # get the name
        name = self.index.get_name(features)

        return name

    def extract_features(self, image):
        # Preprocess the image
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32')
        image = np.expand_dims(image, axis=0)

        # Normalize the image
        image[..., 0] -= 91.4953
        image[..., 1] -= 103.8827
        image[..., 2] -= 131.0912

        # Extract the features
        features = self.vgg_features.predict(image)

        return features
