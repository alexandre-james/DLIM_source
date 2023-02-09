import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN

class FaceClipping:
    def __init__(self):
        # Create the detector
        self.detector = MTCNN()

        self.cropped_faces = {}
        self.bounding_boxes = {}

    def get_faces(self, image, faces):
        cropped_faces = []
        bounding_boxes = []
        if len(faces) > 0:
            # Draw a rectangle around the faces
            # for face in faces:
            #     x, y, w, h = face['box']
            #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            height, width, channels = image.shape
            for i in range(len(faces)):    
                (x, y, w, h) = faces[i]['box']
                center_x = x+w/2
                center_y = y+h/2
                
                b_dim = min(max(w,h)*1.2,width, height)
                box = [center_x-b_dim/2, center_y-b_dim/2, center_x+b_dim/2, center_y+b_dim/2]
                box = [int(x) for x in box]
                # Crop Image
                if box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height:
                    
                    crpim = image[box[1]:box[3],box[0]:box[2]]
                    crpim = cv2.resize(crpim, (224,224), interpolation = cv2.INTER_AREA)
                    #print("Found {0} faces!".format(len(faces)))
                    cropped_faces.append(crpim)
                    bounding_boxes.append(box)

        return cropped_faces, bounding_boxes

    # detect faces for an image
    def detect_image(self, image):
        # detect faces in the image
        faces = self.detector.detect_faces(image)

        # save faces
        cropped_faces, bounding_boxes = self.get_faces(image, faces)

        return cropped_faces, bounding_boxes

    # detect faces for an image
    def detect_file(self, filename):
        # load image from file
        image = cv2.imread(filename)
        self.detect_image(image)
    
    # detect faces for all images in a folder and subfolders
    def detect_folder(self, folder):
        for root, dirs, files in os.walk(folder):
            for filename in files:
                # create the full input path and read the file
                input_path = os.path.join(root, filename)
                image = cv2.imread(input_path)

                # detect faces in the image
                cropped_faces, bounding_boxes = self.detect_image(image)

                name = filename.split('/')[-1].split('.')[0]

                self.cropped_faces[name] = cropped_faces
                self.bounding_boxes[name] = bounding_boxes

    # plot the detected faces of the first image
    def plot(self):
        # get the number of faces of the first image
        nb_faces = len(self.cropped_faces.items()[0][1])
        rows, cols = 3, 3
        for i in range(nb_faces):
            plt.subplot(rows,cols,i+1)
            # show RGB image
            plt.imshow(cv2.cvtColor(self.cropped_faces.items()[0][i], cv2.COLOR_BGR2RGB))
        plt.show()

    # dump the detected faces
    def dump(self, folder):
        # create folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        # dump all faces with the same name as the image and a _ followed by the face number if there are multiple faces
        for name, faces in self.cropped_faces.items():
            if len(faces) == 1:
                cv2.imwrite(folder + '/' + name + '.jpg', faces[0])
            else:
                for i in range(len(faces)):
                    cv2.imwrite(folder + '/' + name + '_' + str(i+1) + '.jpg', faces[i])