import cv2
import numpy as np
import concurrent.futures

from face_clipping import FaceClipping
from classifier import Classifier

# Def result class with results, confidence and bounding boxes
class Result:
    def __init__(self):
        self.results = None
        self.confidence = None
        self.bounding_boxes = None

def draw(image, result):
    # draw the bounding box and label on the image
    if result is None:
        return image

    for i in range(len(result.results)):
        # color is green if the confidence is above 0.85, orange if it is above 0.75 and red otherwise
        if result.confidence[i] > 0.85:
            color = (0, 255, 0)
        elif result.confidence[i] > 0.75:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        (x1, y1, x2, y2) = result.bounding_boxes[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # draw if confidence is above 0.75
        if result.confidence[i] > 0.75:
            # name over lastname
            cv2.putText(image, result.results[i].split('_')[0], (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            # lastname under name
            cv2.putText(image, result.results[i].split('_')[1], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            # confidence under the box
            cv2.putText(image, str(round(result.confidence[i] * 1000) / 10) + "%", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # else draw unknown
        else:
            cv2.putText(image, 'Unknown', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    
    return image

def process(image, classifier, face_clipping):
    # Detect faces
    cropped_faces, bounding_boxes = face_clipping.detect_image(image)

    # if no faces are detected, return None
    if len(cropped_faces) == 0:
        return None

    # Create the result object
    result = Result()

    # Classify the faces
    results, confidence = classifier.classify(cropped_faces)

    # Save the results
    result.results = results
    result.confidence = confidence
    result.bounding_boxes = bounding_boxes

    return result

# Open the camera and detect faces
def camera_vision():
    # Create the classifier
    classifier = Classifier()

    # Load the model
    classifier.load_model()

    # Create the face clipping
    face_clipping = FaceClipping()

    # Create the result object
    result = None

    # Get the image from the camera and process it in a thread

    with concurrent.futures.ThreadPoolExecutor() as executor:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        if not ret:
            cap.release()
            cv2.destroyAllWindows()

        future = executor.submit(process, frame, classifier, face_clipping)

        while(True):
            # Process the image
            if future.done():
                result = future.result()
                future = executor.submit(process, frame, classifier, face_clipping)

            frame = draw(frame, result)

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # Exit if q is pressed
            if cv2.waitKey(1) == ord('q'):
                break

            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def labelize(filename):
    # Create the classifier
    classifier = Classifier()

    # Load the model
    classifier.load_model()

    # Create the face clipping
    face_clipping = FaceClipping()

    # Load the image
    image = cv2.imread(filename)

    # Process the image
    result = process(image, classifier, face_clipping)

    # Draw the image
    image = draw(image, result)

    # Display the image
    cv2.imshow('frame', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    camera_vision()