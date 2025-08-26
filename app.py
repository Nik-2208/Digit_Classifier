import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import joblib

st.title("Handwritten Digit Classifier")

model = joblib.load("knn_model.pkl")

def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)
    img_array = np.array(image).flatten().reshape(1, -1)
    img_array = img_array / 255.0
    return img_array, image

uploaded_file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array, processed_image = preprocess_image(image)
    st.image(processed_image, caption="Processed Image", use_container_width=True)

    prediction = model.predict(img_array)
    st.write(f"Predicted digit: {prediction[0]}") 
