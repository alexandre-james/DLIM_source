import numpy as np
import cv2
from keras_vggface.vggface import VGGFace

class Comparator:
    def __init__(self):        
        # Loading convolution features
        self.vgg_features = VGGFace(
            model='resnet50',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg')
            
        # Define the threshold for the distance
        self.threshold = 100

    # Extract the features from the image
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

    # Compare two images
    def compare(self, image1, image2):
        # Extract the features
        features1 = self.extract_features(image1)
        features2 = self.extract_features(image2)

        # Calculate the distance
        distance = np.linalg.norm(features1 - features2)

        return distance < self.threshold
    
    def compare_features_img(self, features1, image2):
        # Extract the features
        features2 = self.extract_features(image2)

        # Calculate the distance
        distance = np.linalg.norm(features1 - features2)
        print(distance)
        return distance < self.threshold