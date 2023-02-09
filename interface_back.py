
import cv2
from comparator import Comparator
from face_clipping import FaceClipping
from classifier import Classifier

# Search using the reference image all images containing the person
def get_photos_from_face(ref_photo_path, all_photo_path) : 
    good_photos_path = []
    
    # --- Prepare all the references data
    ref_image = cv2.imread(ref_photo_path)
    
    # Be sure we have a good crop of the face in the image
    clip = FaceClipping()
    clip.detect_image(ref_image)
    if len(clip.cropped_faces) != 0 :
        ref_image = clip.cropped_faces[0]
    comparator = Comparator()
    ref_features = comparator.extract_features(ref_image)

    # --- Search for the ref image in all images
    for path in all_photo_path :
        #print(path)
        act_img = cv2.imread(path)
        if act_img.data == None : 
            continue
        cropped_faces, _ = clip.detect_image(act_img)
        for face in cropped_faces :
            if comparator.compare_features_img(ref_features, face) :
                good_photos_path.append(path)
                break

    return good_photos_path; 

# Search using the name all images containing the person
# work only for pre-trained people
def get_photos_from_name(people_name, all_photo_path, classifier: Classifier) :
    good_photos_path = []
    print(people_name)

    clip = FaceClipping()
    threshold_confidence = 0.85

    for path in all_photo_path :
        act_img = cv2.imread(path)
        if act_img.data == None : 
            continue
        cropped_faces, _ = clip.detect_image(act_img)
        if len(cropped_faces) == 0 :
            continue
        class_names, confidence = classifier.classify(cropped_faces)
        for i in range(len(class_names)) : 
            if class_names[i] == people_name and confidence[i] > threshold_confidence : 
                good_photos_path.append(path)
                break


    return good_photos_path
