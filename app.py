import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
MODEL_PATH = "models/asl_cnn.keras"  # or .h5 if you prefer
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'del', 'nothing', 'space'
]

# UI setup
st.set_page_config(page_title="Real-Time Sign Language Translator", layout="wide")
st.title("🤟 Real-Time Sign Language Translator")
st.write("Translate ASL hand signs into text instantly using your trained CNN model!")

# Sidebar options
mode = st.sidebar.radio("Choose Input Mode", ["📷 Live Camera", "🖼️ Upload Image"])

if mode == "📷 Live Camera":
    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    st.markdown("Press **Stop Camera** to end live detection.")

    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Camera not detected!")
            break

        # Preprocess frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(img, (64, 64))
        roi = roi / 255.0
        roi = np.expand_dims(roi, axis=0)

        # Predict
        preds = model.predict(roi)
        pred_class = CLASSES[np.argmax(preds)]

        # Draw box & label
        cv2.rectangle(frame, (50, 50), (400, 400), (0, 255, 0), 2)
        cv2.putText(frame, f"Prediction: {pred_class}", (60, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    camera.release()
    st.success("Camera Stopped.")

elif mode == "🖼️ Upload Image":
    uploaded_file = st.file_uploader("Upload an ASL hand sign image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

        # Preprocess image
        img = image.resize((64, 64))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        preds = model.predict(img)
        pred_class = CLASSES[np.argmax(preds)]

        st.subheader(f"Predicted Sign: 🧠 {pred_class}")
