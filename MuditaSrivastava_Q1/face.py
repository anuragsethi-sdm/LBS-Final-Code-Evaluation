#streamlit for ui 
#impoting libraries
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

#setting title of page 

st.set_page_config(page_title="Face Mask Detection", layout="centered")

st.title(" Face Mask Detection App")
#uploader to upload img

uploaded_file = st.file_uploader("Upload image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model("face_mask_model.h5")
    img = image.resize((100, 100))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    label = np.argmax(prediction)
    class_names = ["With Mask", "Without Mask"]

    st.subheader(f"Prediction: **{class_names[label]}**")