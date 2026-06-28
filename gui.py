import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model("cats_vs_dogs_model.h5")
IMG_SIZE = (150, 150)  # same as training

# Create main window
window = tk.Tk()
window.title("Cat vs Dog Classifier")
window.geometry("400x500")
window.configure(bg="#f0f0f0")

# Title label
title_label = tk.Label(window, text="Cat vs Dog Classifier", font=("Arial", 18, "bold"), bg="#f0f0f0")
title_label.pack(pady=20)

# Function to upload and predict
def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    # Open image for display
    display_img = Image.open(file_path).resize((200, 200))
    img_tk = ImageTk.PhotoImage(display_img)

    # Display image
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # Preprocess image for prediction
    img = display_img.resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array.reshape((1, IMG_SIZE[0], IMG_SIZE[1], 3))
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)[0][0]
    result = "Dog" if prediction >= 0.5 else "Cat"

    # Show result
    result_label.config(text=f"Prediction: {result}")

# Upload button
upload_btn = tk.Button(window, text="Upload Image", command=upload_and_predict,
                       font=("Arial", 14), bg="#4CAF50", fg="white", width=20)
upload_btn.pack(pady=10)

# Image display label
image_label = tk.Label(window, bg="#f0f0f0")
image_label.pack(pady=10)

# Prediction result label
result_label = tk.Label(window, text="", font=("Arial", 14, "bold"), fg="blue", bg="#f0f0f0")
result_label.pack(pady=10)

# Run GUI
window.mainloop()
