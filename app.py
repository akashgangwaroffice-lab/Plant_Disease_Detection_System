import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from disease_info import disease_info

MODEL_PATH = "models/plant_disease_model.keras"

class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

model = tf.keras.models.load_model(MODEL_PATH)

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱"
)

st.title("🌱 Plant Disease Detection System")

st.write(
    "Upload a leaf image to detect disease."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )

    img = image.resize((224, 224))

    img_array = np.array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction) * 100

    st.success(
    f"Disease Detected: {disease_info[predicted_class]['display_name']}"
)

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    st.subheader("Description")

    st.write(
        disease_info[predicted_class]["description"]
    )

    st.subheader("Symptoms")
    st.write(
        disease_info[predicted_class]["symptoms"]
    )

    st.subheader("Recommended Cure")
    st.write(
        disease_info[predicted_class]["cure"]
    )

    st.subheader("Prevention")
    st.write(
        disease_info[predicted_class]["prevention"]
    )