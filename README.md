# DLIM source

On this repo you can find public source files of the DLIM project.
However, you can't run the code due to the removal of pretrained model and data for privacy reasons.
Thank you for your understanding.

## Introduction

This project is about identify people in a image with an application like we see in photo gallery.

## Presentation

The slides from the presentation of the 05/25/2022

[Link to the slides][presentation]

## Team

Antoine Aubin,
Alexandre James,
Héloïse Fabre,
Thibaut Ambrosino,

## Files

```
|- classifier.py
|- data_extraction.py
|- index.py
|- main.py
|- face_clipping.py
|- comparator.py
|- interface.py
|- interface_back.py
|- image_classifier.py
|- vision.py
|- labels.csv
|- data 
	|- CRI
	|- Linkedin
	|- Ephemere
	|- Ephemere2
	|- html
```

## Pipeline

	1. data extraction : get, parse and train on data from CRI
	2. comparator : compare images to know if they are similar
	3. classification : create model from images of CRI, Linkedin and Instagram
	4. face clipping : split test images (Ephemere) into faces to be classified
	5. labelize : give a name with the classifier to those new faces to identify the person

	Interface -> launch the application

## Datas
Training and validation data : EPITA CRI, LinkedIn & Instagram photos of students

_The model was only trained on ING3 SIGL and ING3 IMAGE students_

## How to launch the application:

```python
	python interface.py #Launch the main interface
	# For the search by name, write the name in the format alexandre_james

	python vision.py #Launch the realtime recognition on the webcam
```

## Division of labor


```
Alexandre: Comparator, Classification, Real-time video
Héloise: Data extraction, Labeling
Thibaut: Labeling, Face cutting, Data extraction
Antoine: Interface, Comparator
```

[presentation]: https://github.com/alexandre-james/DLIM_source/raw/main/docs/presentation.pdf
