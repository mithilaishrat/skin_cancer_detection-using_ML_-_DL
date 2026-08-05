import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ============================================================
# Skin Cancer Detection - CNN Streamlit Application
# Based on the uploaded notebook:
# skin_cancer_DL(3).ipynb
# ============================================================

MODEL_PATH = "skin_cancer_cnn.h5"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------
@st.cache_resource
def load_skin_cancer_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        return None


model = load_skin_cancer_model()

# ------------------------------------------------------------
# Page title
# ------------------------------------------------------------
st.title("🩺 Skin Cancer Detection System")
st.write(
    "Upload a skin lesion image to classify it as "
    "**Benign (Non-Cancer)** or **Malignant (Cancer)**."
)

st.info(
    "This application uses the CNN model trained in the uploaded "
    "notebook. The model expects images resized to 224 × 224 pixels "
    "and normalized to the range [0, 1]."
)

# ------------------------------------------------------------
# Model status
# ------------------------------------------------------------
if model is None:
    st.error(
        f"Model file '{MODEL_PATH}' was not found or could not be loaded. "
        "Place the trained model file in the same folder as app.py."
    )
    st.stop()

st.success("CNN model loaded successfully.")

# ------------------------------------------------------------
# Image uploader
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open uploaded image
    original_image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(original_image, caption="Selected Image", use_container_width=True)

    # --------------------------------------------------------
    # Prediction button
    # --------------------------------------------------------
    if st.button("🔍 Predict", use_container_width=True):

        with st.spinner("Analyzing image..."):

            # Resize to the same size used by the notebook
            img = original_image.resize(IMG_SIZE)

            # Convert image to NumPy array and normalize
            img_array = np.asarray(img, dtype=np.float32) / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            # Model prediction
            prediction = model.predict(img_array, verbose=0)

            probability = float(np.asarray(prediction).squeeze())

            # The notebook uses:
            # prediction > 0.5 -> Malignant
            # prediction <= 0.5 -> Benign
            if probability > 0.5:
                class_label = "Malignant"
                confidence = probability * 100
            else:
                class_label = "Benign"
                confidence = (1 - probability) * 100

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------
        st.subheader("Prediction Result")

        if class_label == "Malignant":
            st.error(f"⚠️ Prediction: {class_label}")
        else:
            st.success(f"✅ Prediction: {class_label}")

        st.metric("Confidence", f"{confidence:.2f}%")

        # Show model probability
        st.write(f"Malignant probability: **{probability * 100:.2f}%**")
        st.write(f"Benign probability: **{(1 - probability) * 100:.2f}%**")

        st.warning(
            "This system is for educational/research purposes only and "
            "is not a medical diagnosis. Please consult a qualified "
            "healthcare professional for clinical evaluation."
        )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption(
    "Skin Cancer Detection using CNN | Image size: 224 × 224 | "
    "Binary classification: Benign vs Malignant"
)
