import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.title("Cat vs Dog Classifier")
st.write("Upload a photo of a cat or a dog and the model will classify it.")

# Load the trained model (cached so it only loads once per session)
@st.cache_resource
def get_model():
    return load_model("cats_vs_dogs_model.h5")

model = get_model()
IMG_SIZE = (150, 150)  # same as training

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    # Preprocess image for prediction (same steps as original gui.py)
    img = image.resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array.reshape((1, IMG_SIZE[0], IMG_SIZE[1], 3))
    img_array = img_array / 255.0

    # Predict
    prediction = float(model.predict(img_array)[0][0])
    label = "Dog" if prediction >= 0.5 else "Cat"
    confidence = prediction if label == "Dog" else 1 - prediction

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence * 100:.1f}%")
