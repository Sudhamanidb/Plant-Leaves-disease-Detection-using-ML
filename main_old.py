import tkinter as tk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import imutils
import os

# Disease labels
li = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

model_path = r'C:\Users\Sudha\MAIN PROJECT\plant disease detection\final changed without cam\PLANT_MODEL.h5'

# Load the model safely
try:
    classifier = load_model(model_path, compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    exit()

# Initialize GUI
root = tk.Tk()
root.title("Plant Disease Detector")
root.geometry("800x600")
root.configure(background="gray25")

title = tk.Label(text="Select An Image To Process", background="gray25", fg="white", font=("helv36", 15))
title.grid(row=0, column=2, padx=10, pady=10)

# Global variables
diseasename = None
path = None

# GUI functions
def exit_app():
    root.destroy()

def clear():
    cv2.destroyAllWindows()
    disease = tk.Label(text=' ' * 50, background="gray25", fg="Black", font=("", 20))
    disease.grid(column=3, row=3, padx=10, pady=10)

def classify():
    global path, diseasename
    path = askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not path:
        return

    try:
        img = image.load_img(path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = classifier.predict(img_array)
        pred_index = np.argmax(prediction)
        diseasename = li[pred_index]

        pred_label = tk.Label(text="Prediction: " + diseasename, background="gray25", fg="white", font=("Helvetica", 18))
        pred_label.grid(column=2, row=3, padx=10, pady=10)

        show_image(path)
    except Exception as e:
        error_label = tk.Label(text=f"Prediction Error: {e}", background="gray25", fg="red", font=("Helvetica", 12))
        error_label.grid(column=2, row=3, padx=10, pady=10)

def show_image(image_path):
    img = cv2.imread(image_path)
    img = imutils.resize(img, width=400)
    cv2.imshow("Selected Leaf", img)

# Buttons
btn = tk.Button(root, text="Select Image", command=classify)
btn.grid(column=2, row=1, padx=10, pady=10)

exit_btn = tk.Button(root, text="Exit", command=exit_app)
exit_btn.grid(column=2, row=5, padx=10, pady=10)

clear_btn = tk.Button(root, text="Clear", command=clear)
clear_btn.grid(column=2, row=4, padx=10, pady=10)

root.mainloop()
