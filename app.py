import streamlit as st
import numpy as np
import cv2
import librosa
from tensorflow.keras.models import load_model
from PIL import Image
from audiorecorder import audiorecorder

st.set_page_config(page_title="Parkinson Detection AI", layout="wide")

# -----------------------------
# Load Models (cached)
# -----------------------------
@st.cache_resource
def load_models():
    speech_model = load_model("fusion_model.keras")
    spiral_model = load_model("spiral_model.h5")
    handwriting_model = load_model("handwriting_model.keras")
    return speech_model, spiral_model, handwriting_model

speech_model, spiral_model, handwriting_model = load_models()

# -----------------------------
# Title
# -----------------------------
st.title("🧠 AI-Based Parkinson’s Multi-Modal Detection System")
st.subheader("Speech + Spiral + Handwriting Analysis")

st.write("Upload or record inputs to analyze Parkinson's disease symptoms.")

# -----------------------------
# Speech Analysis
# -----------------------------
st.header("🎤 Speech Analysis")

audio = audiorecorder("Click to record", "Recording...")

if len(audio) > 0:
    st.audio(audio.export().read())

    y, sr = librosa.load(audio.export(), sr=22050)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    mfcc = np.expand_dims(mfcc, axis=0)

    prediction = speech_model.predict(mfcc)

    if prediction[0][0] > 0.5:
        st.error("⚠️ Parkinson symptoms detected in speech")
    else:
        st.success("✅ Speech appears normal")

# -----------------------------
# Spiral Drawing Detection
# -----------------------------
st.header("🌀 Spiral Drawing Analysis")

spiral_file = st.file_uploader("Upload Spiral Image", type=["jpg", "png", "jpeg"])

if spiral_file is not None:

    image = Image.open(spiral_file)
    st.image(image, caption="Uploaded Spiral", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = spiral_model.predict(img)

    if prediction[0][0] > 0.5:
        st.error("⚠️ Parkinson symptoms detected in spiral drawing")
    else:
        st.success("✅ Spiral drawing appears normal")

# -----------------------------
# Handwriting Detection
# -----------------------------
st.header("✍️ Handwriting Analysis")

handwriting_file = st.file_uploader("Upload Handwriting Image", type=["jpg","png","jpeg"])

if handwriting_file is not None:

    image = Image.open(handwriting_file)
    st.image(image, caption="Uploaded Handwriting", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = handwriting_model.predict(img)

    if prediction[0][0] > 0.5:
        st.error("⚠️ Parkinson symptoms detected in handwriting")
    else:
        st.success("✅ Handwriting appears normal")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("AI Parkinson Detection System | Multi-Modal ML Analysis")
