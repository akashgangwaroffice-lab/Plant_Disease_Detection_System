import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
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

img_path = "test_images/test.jpg"

img = image.load_img(
    img_path,
    target_size=(224, 224)
)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)

predicted_class = class_names[predicted_index]

confidence = np.max(prediction) * 100

print(f"\nDisease: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")

print("\nDescription:")
print(disease_info[predicted_class]["description"])

print("\nRecommended Cure:")
print(disease_info[predicted_class]["cure"])