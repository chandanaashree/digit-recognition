import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Digit Recognition", layout="centered")

st.title("Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (0–9)")

# Load model
model = load_model("cnn_model.keras")


def preprocess_image(image):
    img = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Invert colors (MNIST format)
    gray = cv2.bitwise_not(gray)

    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28))

    # Normalize
    normalized = resized / 255.0

    # Reshape for CNN
    reshaped = normalized.reshape(1, 28, 28, 1)

    return reshaped, resized


uploaded_file = st.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width=200)

    processed_image, resized_image = preprocess_image(image)

    with col2:
        st.subheader("Processed (28×28)")
        st.image(resized_image, width=200, clamp=True)

    prediction = model.predict(processed_image)
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"### Predicted Digit: {digit}")
    st.info(f"Confidence: **{confidence:.2f}%**")
