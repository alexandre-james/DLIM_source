import os
import cv2

from comparator import Comparator
from image_classifier import ImageClassifier
from classifier import Classifier
from face_clipping import FaceClipping
from data_extraction import get_crifaces, delete_images, get_dataset
from index import Indexer
from vision import camera_vision, labelize

# Compare all images
def compare_all_images():
    # Load all the images in the folder
    images_path = 'data/images/'
    images_filenames = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    images = [cv2.imread(images_path + f) for f in images_filenames]

    # Create the comparator
    comparator = Comparator()

    # Compare the images
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            if comparator.compare(images[i], images[j]):
                print(images_filenames[i] + ' is similar to ' + images_filenames[j])

# Create, train and save the model
def create_model():
    # Create the classifier
    classifier = ImageClassifier('data/CRI/')

    # Train the model
    classifier.train_model()

# Load the model and classify the images
def classify_images():
    # Create the classifier
    classifier = Classifier()

    # Train the model
    classifier.train_model(['data/CRI/', 'data/Linkedin/'])

    # Get the validation dataset
    images, labels = get_dataset('data/Ephemere/labels/')
    found = 0
    recognized = 0
    recognized_found = 0

    results, confidence = classifier.classify(images)

    for i in range(len(results)):
        if confidence[i] > 0.75:
            found += 1
        if results[i] == labels[i]:
            recognized += 1
            if confidence[i] > 0.85:
                recognized_found += 1

    accuracy = recognized / len(results)
    found_accuracy = found / len(results)
    recognized_found_accuracy = recognized_found / found

    print('Accuracy: ' + str(accuracy))
    print('Found accuracy: ' + str(found_accuracy))
    print('Recognized found accuracy: ' + str(recognized_found_accuracy))

def index_images():
    # Create the indexer
    indexer = Indexer('data/CRI/IMAGE/')

    # Print the index
    indexer.index.pretty_print()

def create_dataset():
    for classname, length in [('IMAGE', 52), ('SIGL', 58)]:
        get_crifaces(classname)
        delete_images(classname, length)

def clip_faces():
    face_clipping = FaceClipping()
    face_clipping.detect_folder('data/Ephemere 2/raw/')
    face_clipping.dump('data/Ephemere/labels/')

def __main__():
    #print(get_dataset('data/Ephemere/labels/'))
    labelize("data/Ephemere/raw/9.jpg")
    
    #classify_images()
    #create_dataset()
    #camera_vision()

if __name__ == '__main__':
    __main__()