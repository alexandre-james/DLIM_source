import tkinter as tk
from tkinter import *
from tkinter import filedialog
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from PIL import Image, ImageTk
import functools
from interface_back import *
import shutil
from classifier import Classifier
from face_clipping import FaceClipping
from vision import process, draw

photos_path = os.getcwd()
main_window: tk.Tk = None
text_path_area: tk.Text = None
reference_image_path = ""
results_path = "results0"
name_input = None
# Load the classifier
classifier = Classifier()
classifier.load_model()

def get_tk_image_from_image(image, shape, shapeIsMax = True) :
    # Respect the ratio with size limit
    if shapeIsMax : 
        shape_ratio = max(shape[0]/image.width, shape[1]/image.height)
        shape = (round(image.width * shape_ratio), round(image.height * shape_ratio))
    image = image.resize(shape)
    image = ImageTk.PhotoImage(image)
    return image

def get_tk_image(path, shape, shapeIsMax = True) : 
    image = Image.open(path, 'r')
    return get_tk_image_from_image(image, shape, shapeIsMax)

def launch_command(seeResultsButton, use_name = False) :
    global reference_image_path
    global photos_path
    global results_path
    global name_input

    print("Launched research !")

    all_photos_path = get_photos_path(photos_path)
    if len(all_photos_path) == 0 : 
        print("No photos found in selected directory")
        return
    
    if not use_name :
        all_results_path = get_photos_from_face(reference_image_path, all_photos_path)
    else : 
        all_results_path = get_photos_from_name(name_input.get(), all_photos_path, classifier)
    print(len(all_results_path), " photos found with this person")
    
    free_path = False
    while not free_path : 
        if not os.path.exists(results_path) :
            os.mkdir(results_path)
            free_path = True
        else :
            results_path += '_'
    
    for p in all_results_path :
        end_path = p.split('/')[-1]
        shutil.copy(p, os.path.join(results_path, end_path))

    # When done, enable the see results button
    seeResultsButton["state"] = "active"

def set_result_image(label, path) :
    image_display = get_tk_image(path, (300, 300), True)
    label.configure(image=image_display)
    label.image = image_display

def see_all_resultsCommand() : 
    global results_path
    # Open a window with all the images contained in "result folder" 
    result_window = tk.Toplevel(main_window); 
    result_window.title("Results")
    result_window.geometry("900x350")
    resultleft_panel = tk.PanedWindow(result_window, bd=4, relief="raised", orient="vertical")
    resultleft_panel.pack(expand=1, side="left")
    resultright_paned = tk.PanedWindow(result_window, bd=2, relief="raised", orient="vertical")
    resultright_paned.pack(expand=1, side="right")

    frame_container=Frame(resultright_paned)

    canvas_container=Canvas(frame_container, height=250)
    frame2=Frame(canvas_container)
    myscrollbar=Scrollbar(frame_container,orient="vertical",command=canvas_container.yview) # will be visible if the frame2 is to to big for the canvas
    canvas_container.create_window((0,0),window=frame2,anchor='nw')

    all_results_path = get_photos_path(results_path)

    # Image displayed
    image_display = get_tk_image(all_results_path[0], (300, 300), True)
    label_img = tk.Label(result_window, text="Image", image=image_display)
    resultleft_panel.add(label_img)

    # Create all the buttons
    for i in range(len(all_results_path)) : 
        splitted_path = all_results_path[i].split('/')
        button = tk.Button(frame2, text=splitted_path[-1], command=functools.partial(set_result_image,label_img, all_results_path[i]))
        button.pack(expand=2)

    frame2.update() 

    canvas_container.configure(yscrollcommand=myscrollbar.set, scrollregion="0 0 0 %s" % frame2.winfo_height())
    canvas_container.pack(side=LEFT)
    myscrollbar.pack(side=RIGHT, fill = Y)

    frame_container.pack()

    result_window.mainloop()

# Search all recursively all pictures from given path
def get_photos_path(path) :
    all_photos_path = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                full_path = os.path.join(root, file)
                all_photos_path.append(full_path)
    return all_photos_path

def set_path(text_path_area) : 
    global photos_path
    new_path = tk.filedialog.askdirectory(parent=main_window, title="Select the photos' parent folder")
    if new_path != None : 
        photos_path = new_path
        text_path_area.config(text="Will search photos in " + photos_path)

def set_reference_image(label) :
    global reference_image_path
    new_path = tk.filedialog.askopenfile(parent=main_window, title="Choose reference picture")
    if new_path != None : 
        reference_image_path = new_path.name
        #print("User set a new reference path ", reference_image_path)
        image = get_tk_image(reference_image_path, (100, 200), True)
        label.configure(image=image)
        label.image = image

def tell_people_command(ref_photo_path) : 
    face_clipping = FaceClipping()
    
    frame = cv2.imread(ref_photo_path)
    # Get a picture with the name of peoples in it
    result = process(frame, classifier, face_clipping)
    res_image = draw(frame, result)

    res_filename = ref_photo_path.split('.')[0] + "_labelized.jpg"
    cv2.imwrite(res_filename, res_image)

    result_window = tk.Toplevel(main_window); 
    result_window.title("Results")
    result_window.geometry("1200x800")
    image = get_tk_image(res_filename, (1200, 800))
    label_img = tk.Label(result_window, image=image)
    label_img.pack()
    result_window.mainloop()

def main() : 
    global main_window
    global reference_image_path
    global name_input

    main_window = tk.Tk()
    main_window.title("Face finder !")
    main_window.geometry("800x400")

    # --- Left part, inputs

    left_panel = tk.PanedWindow(orient="vertical")
    left_panel.pack(expand=1, side="left")

    reference_image_path = ""
    image = None#get_tk_image(reference_image_path, (100, 200))
    label_img = tk.Label(main_window, text="Reference Image\n\n\n\n\n\n\n\n", image=image)
    set_path_ref_button = tk.Button(main_window, text="Select reference image", command = lambda: set_reference_image(label_img))
    left_panel.add(set_path_ref_button)
    left_panel.add(label_img)
    name_input = StringVar()
    label_input = tk.Label(main_window, text="Reference name")
    left_panel.add(label_input)
    name_input_zone = tk.Entry(main_window, textvariable=name_input)
    name_input_zone.focus_set()
    left_panel.add(name_input_zone)


    # --- Right part, results

    right_paned = tk.PanedWindow(orient="vertical")
    right_paned.pack(side="right")

    title_area = tk.Label(main_window, text="Search picture containing the reference:", font = "Verdana 10 bold")
    right_paned.add(title_area)

    text_path_area = tk.Label(main_window, text="Will search photos in " + photos_path)
    right_paned.add(text_path_area)

    see_all_results = tk.Button(main_window, text="See all results", command = see_all_resultsCommand)
    see_all_results["state"] = "disable"

    set_pathButton = tk.Button(main_window, text="Select research folder", command = lambda: set_path(text_path_area))
    right_paned.add(set_pathButton)
    launch_button = tk.Button(main_window, text="Research using image", command = lambda: launch_command(see_all_results))
    right_paned.add(launch_button)
    launch_button2 = tk.Button(main_window, text="Research using name", command = lambda: launch_command(see_all_results, True))
    right_paned.add(launch_button2)
    right_paned.add(see_all_results)

    title_area2 = tk.Label(main_window, text="Tell who is in the reference image:", font = "Verdana 10 bold")
    right_paned.add(title_area2)
    label_people_result = tk.Label(main_window, text="")
    tell_people_button = tk.Button(main_window, text="Tell names", command = lambda: tell_people_command(reference_image_path))
    right_paned.add(tell_people_button)
    right_paned.add(label_people_result)

    main_window.mainloop()

main()