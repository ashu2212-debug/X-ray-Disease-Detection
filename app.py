import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load Model
model = load_model("xray_disease_model.h5")

classes = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TURBERCULOSIS"
]

st.title("Chest X-Ray Disease Detection")

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", width=300)

    img = img.resize((224,224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")