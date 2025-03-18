import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import gdown

url = "https://drive.google.com/file/d/1V7Aqjnmy3rgZYVIK9XthP0nTVmIgeHnx/view?usp=sharing"
output = "ResNet50.h5"
gdown.download(url, output, quiet=False)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)   
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("ResNet50.h5")  
    return model

model = load_model()

def predict_wheat_quality(image):
    img = image.resize((224, 224))  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    prediction = model.predict(img_array)  
    probability = prediction[0][0] 

    class_labels = ["Damaged Grain", "Fine Grain"]  
    result = class_labels[int(probability > 0.5)] 
    return result, probability

st.title("🌾 Wheat Quality Analysis")
st.write("Upload an image of wheat grains to determine their quality.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((250, 250))
    st.image(image, caption="Uploaded Image")

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        result, confidence = predict_wheat_quality(image)
        st.success(f"Prediction: {result}")
