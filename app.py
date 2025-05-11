import streamlit as st
import tensorflow as tf
import numpy as np
from numpy.core.defchararray import center


def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


st.set_page_config(page_title="Plant Disease Recognition", page_icon="🌿", layout="wide")
st.markdown(
    """
    <h1 style="text-align: center; color: white;">Plant Disease Identifier</h1>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<h2 style='text-align: left; color: white;'>Machine Learning Project</h2>", unsafe_allow_html=True)

st.sidebar.markdown("<h3 style='text-align: left; color: white;'>Team Members</h3>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <p style="font-size: 16px; color: white; text-align: left;">
        <b>Muhammad Ahsan</b> (21-SE-62)<br>
        <b>Talha Mushtaq</b> (21-SE-02)
    </p>
    <br>
    """,
    unsafe_allow_html=True
)
app_mode = st.sidebar.radio("Choose an option", ["Main", "About Dataset"])

if app_mode == "Main":
    st.markdown(
        '<div class="title-text" style="text-align: center; font-size: 35px; color: #2e7d32;">Upload an Image of Your Plant </div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="description-text" style="text-align: center; font-size: 18px; color: #555;">We will analyze the image and tell you if there is any disease in the plant leaf.</div>',
        unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    test_image = st.file_uploader("Choose an image of the plant leaf:", type=["jpg", "jpeg", "png"],
                                  key="image_uploader")

    if test_image:
        st.image(test_image, caption="Uploaded Image", use_container_width=False, width=300,)
        if st.button("Predict Disease"):
            with st.spinner('Analyzing...'):
                result_index = model_prediction(test_image)

                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                    'Potato___Late_blight',
                    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                    'Tomato___Early_blight',
                    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                ]

                # Check if the prediction corresponds to a healthy plant
                predicted_class = class_name[result_index]
                if "healthy" in predicted_class:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.success(f"Prediction: This plant is healthy")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.error(f"Prediction: Disease Detected - {predicted_class}")
                    st.markdown('</div>', unsafe_allow_html=True)



elif app_mode == "About Dataset":
    st.header("About the Dataset")
    st.markdown(
        """
        #### Dataset Overview
        This dataset is derived from the [Plant Disease Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

        The dataset consists of approximately **87,000 RGB images** of healthy and diseased crop leaves, categorized into **38 different classes**. 
        These images are organized into training, validation, and testing sets in an **80/20 ratio**, preserving the directory structure.

        #### Content
        - **Training Set:** 70,295 images
        - **Validation Set:** 17,572 images
        - **Test Set:** 33 images (created separately for prediction purposes)

        #### Key Features
        - Helps identify diseases in crop leaves.
        - Useful for building machine learning models for plant disease detection.

        ---

        ### Dataset Reference
         [Kaggle dataset page](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).
        """
    )

