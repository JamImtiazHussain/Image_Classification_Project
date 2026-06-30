import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model (same model file used by the original Tkinter GUI)
model = load_model("cats_vs_dogs_model.h5")
IMG_SIZE = (150, 150)  # same as training


def predict(image):
    if image is None:
        return "Please upload an image."

    # Preprocess image for prediction (same steps as original gui.py)
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array.reshape((1, IMG_SIZE[0], IMG_SIZE[1], 3))
    img_array = img_array / 255.0

    # Predict
    prediction = float(model.predict(img_array)[0][0])
    label = "Dog" if prediction >= 0.5 else "Cat"
    confidence = prediction if label == "Dog" else 1 - prediction

    return f"{label} ({confidence * 100:.1f}% confidence)"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Cat vs Dog Classifier",
    description="Upload a photo of a cat or a dog and the model will classify it.",
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
