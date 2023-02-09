import os
import cv2
import numpy as np
from bs4 import BeautifulSoup

# Read the images in the source folder and save them in the destination folder
# with the correct name in the format 'name.lastname.jpg'
def get_crifaces(classname):
    source_folder = "data/html/" + classname + "_files/"
    destination_folder = "data/CRI/" + classname + "/"
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpg'):
            try:
                image = cv2.imread(source_folder + filename)
                name = filename.split('_')[0]
                if len(name.split('.')) >= 2:
                    firstname, lastname = name.split('.')
                    cv2.imwrite(destination_folder + firstname + '_' + lastname + '.jpg', image)
            except:
                print('Error with ' + filename)

# Read the html and extract the name and lastname of each person
# and save them in a list
def get_names(classname, length):
    file = "data/html/" + classname + ".html"
    with open(file, 'r') as f:
        soup = BeautifulSoup(f, 'html.parser')
        names = []
        # get the text content of all 'btn btn-link btn-light btn-sm card-body p-3 text-monospace' classes
        for name in soup.find_all('a', class_='btn btn-link btn-light btn-sm card-body p-3 text-monospace'):
            # remove the ' ' and the '\n' characters
            name = name.text.replace(' ', '').replace('\n', '')
            if len(name.split('.')) >= 2:
                firstname, lastname = name.split('.')
                names.append(firstname + '_' + lastname)
        return names[:length]

# Deletes images that are not in the list of names
def delete_images(classname, length):
    names = get_names(classname, length)
    images_path = "data/CRI/" + classname + "/"
    # create path if it doesn't exist
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    images_filenames = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    for filename in images_filenames:
        name = os.path.splitext(filename)[0][0]
        if name not in names:
            os.remove(os.path.join(images_path, filename))

# Get the filenames of all the images in the folder and subfolders
def get_filenames(folder):
    filenames = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.jpg'):
                filenames.append(os.path.join(root, file))
    return filenames

# Create a dataset from the images in the folder
# Images are labeled with the name_lastname of the person
# If there is multiple images of the same person they have _1, _2, _3, ... at the end of the name
# We return two lists: one with the images and one with the labels
def get_dataset(folder):
    filenames = get_filenames(folder)
    images = []
    labels = []
    for filename in filenames:
        image = cv2.imread(filename)
        images.append(image)
        name = os.path.splitext(os.path.basename(filename))[0]
        if len(name.split('_')) >= 2:
            name = name.split('_')[0] + '_' + name.split('_')[1]
        labels.append(name)
    return images, labels
        