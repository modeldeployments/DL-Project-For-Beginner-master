import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import (
    preprocess_input,
    decode_predictions
)


# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(
    page_title="VGG19 Image Classifier",
    page_icon="🖼️"
)


# ---------------------------------
# Load model
# ---------------------------------
@st.cache_resource
def load_vgg19_model():
    return load_model("vgg19.keras")


model = load_vgg19_model()


# ---------------------------------
# Session state for image uploader
# ---------------------------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ---------------------------------
# Title
# ---------------------------------
st.title("🖼️ VGG19 Image Classifier")

st.write(
    "Upload an image and VGG19 will predict the image class."
)


# ---------------------------------
# Upload image
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    key=f"image_uploader_{st.session_state.uploader_key}"
)


# ---------------------------------
# Display uploaded image
# ---------------------------------
if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )


    # ---------------------------------
    # Predict
    # ---------------------------------
    if st.button("🔍 Predict"):

        with st.spinner("Analyzing image..."):

            img = image.load_img(
                uploaded_file,
                target_size=(224, 224)
            )

            x = image.img_to_array(img)

            x = np.expand_dims(x, axis=0)

            x = preprocess_input(x)

            preds = model.predict(
                x,
                verbose=0
            )

            predictions = decode_predictions(
                preds,
                top=3
            )[0]


        # ---------------------------------
        # Results
        # ---------------------------------
        st.subheader("Prediction Results")

        for _, class_name, probability in predictions:

            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

        st.success(
            f"Top prediction: {predictions[0][1]}"
        )


    # ---------------------------------
    # Upload another image
    # ---------------------------------
    st.divider()

    if st.button("🔄 Upload Another Image"):

        # Change uploader key
        st.session_state.uploader_key += 1

        # Rerun application
        st.rerun()