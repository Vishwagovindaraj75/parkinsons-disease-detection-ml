import streamlit as st
import numpy as np
import cv2
import librosa
import os
import tempfile
from tensorflow.keras.models import load_model
from PIL import Image
from audiorecorder import audiorecorder

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(page_title="Parkinson Detection AI", layout="wide")

st.title("🧠 AI-Based Parkinson’s Multi-Modal Detection System")
st.write("Speech + Spiral + Handwriting Analysis")

# ----------------------------
# Show files in directory (debug)
# ----------------------------
st.write("Files available in repo:", os.listdir())

# ----------------------------
# Load models
# ----------------------------
@st.cache_resource
def load_models():

    speech_model = None
    spiral_model = None
    handwriting_model = None

    try:
        speech_model = load_model("fusion_model.keras", compile=False)
        st.success("Speech model loaded")
    except Exception as e:
        st.warning(f"Speech model error: {e}")

    try:
        spiral_model = load_model("spiral_model.h5", compile=False)
        st.success("Spiral model loaded")
    except Exception as e:
        st.warning(f"Spiral model error: {e}")

    try:
        handwriting_model = load_model("handwriting_model.keras", compile=False)
        st.success("Handwriting model loaded")
    except Exception as e:
        st.warning(f"Handwriting model error: {e}")

    return speech_model, spiral_model, handwriting_model


speech_model, spiral_model, handwriting_model = load_models()

# =====================================================
# SPEECH ANALYSIS
# =====================================================
st.header("🎤 Speech Analysis")

audio = audiorecorder("Start Recording", "Stop Recording")

if len(audio) > 0:

    st.audio(audio.export().read())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio.export(f.name, format="wav")
        audio_path = f.name

    y, sr = librosa.load(audio_path, sr=22050)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=0)

    if speech_model is not None:

        prediction = speech_model.predict(mfcc)

        if prediction[0][0] > 0.5:
            st.error("⚠ Parkinson symptoms detected in speech")
        else:
            st.success("✅ Speech appears normal")

# =====================================================
# SPIRAL ANALYSIS
# =====================================================
st.header("🌀 Spiral Drawing Detection")

spiral_file = st.file_uploader(
    "Upload Spiral Image",
    type=["jpg", "jpeg", "png"],
    key="spiral"
)

if spiral_file is not None:

    image = Image.open(spiral_file)
    st.image(image, caption="Uploaded Spiral")

    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    if spiral_model is not None:

        prediction = spiral_model.predict(img)

        if prediction[0][0] > 0.5:
            st.error("⚠ Parkinson symptoms detected in spiral drawing")
        else:
            st.success("✅ Spiral drawing appears normal")

# =====================================================
# HANDWRITING ANALYSIS
# =====================================================
st.header("✍ Handwriting Analysis")

hand_file = st.file_uploader(
    "Upload Handwriting Image",
    type=["jpg", "jpeg", "png"],
    key="handwriting"
)

if hand_file is not None:

    image = Image.open(hand_file)
    st.image(image, caption="Uploaded Handwriting")

    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    if handwriting_model is not None:

        prediction = handwriting_model.predict(img)

        if prediction[0][0] > 0.5:
            st.error("⚠ Parkinson symptoms detected in handwriting")
        else:
            st.success("✅ Handwriting appears normal")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.write("AI Parkinson Detection System | Multi-Modal Machine Learning")
