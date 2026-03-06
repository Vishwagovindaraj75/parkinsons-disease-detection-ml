import streamlit as st
import numpy as np
import cv2
import librosa
import os
import tempfile
import tensorflow as tf
from PIL import Image
from audiorecorder import audiorecorder

# ---------------------------
# Ignore quantization_config issue
# ---------------------------
from tensorflow.keras.layers import Dense as OriginalDense

class PatchedDense(OriginalDense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

custom_objects = {"Dense": PatchedDense}

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Parkinson Detection AI", layout="wide")

st.title("🧠 AI-Based Parkinson’s Multi-Modal Detection System")
st.write("Speech + Spiral + Handwriting Analysis")

# Debug: show repo files
st.write("Files available in repo:", os.listdir())

# ---------------------------
# Load models
# ---------------------------
@st.cache_resource
def load_models():

    speech_model = None
    spiral_model = None
    handwriting_model = None

    try:
        speech_model = tf.keras.models.load_model(
            "fusion_model.h5",
            compile=False,
            custom_objects=custom_objects
        )
        st.success("Speech model loaded")
    except Exception as e:
        st.error(f"Speech model error: {e}")

    try:
        spiral_model = tf.keras.models.load_model(
            "spiral_model.h5",
            compile=False,
            custom_objects=custom_objects
        )
        st.success("Spiral model loaded")
    except Exception as e:
        st.error(f"Spiral model error: {e}")

    try:
        handwriting_model = tf.keras.models.load_model(
            "handwriting_model.h5",
            compile=False,
            custom_objects=custom_objects
        )
        st.success("Handwriting model loaded")
    except Exception as e:
        st.error(f"Handwriting model error: {e}")

    return speech_model, spiral_model, handwriting_model


speech_model, spiral_model, handwriting_model = load_models()

# =====================================================
# SPEECH ANALYSIS
# =====================================================
st.header("🎤 Speech Analysis")

audio = audiorecorder("Start Recording", "Stop Recording")

if len(audio) > 0:

    audio_bytes = audio.export().read()
    st.audio(audio_bytes)

    try:
        # create temp audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name

        # load audio
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

    except Exception as e:
        st.error(f"Audio processing error: {e}")

# =====================================================
# SPIRAL ANALYSIS
# =====================================================
st.header("🌀 Spiral Drawing Detection")

spiral_file = st.file_uploader(
    "Upload Spiral Image",
    type=["jpg", "jpeg", "png"]
)

if spiral_file is not None:

    try:
        image = Image.open(spiral_file)
        st.image(image)

        img = np.array(image)
        img = cv2.resize(img, (128,128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        if spiral_model is not None:

            prediction = spiral_model.predict(img)

            if prediction[0][0] > 0.5:
                st.error("⚠ Parkinson symptoms detected in spiral")
            else:
                st.success("✅ Spiral looks normal")

    except Exception as e:
        st.error(f"Spiral processing error: {e}")

# =====================================================
# HANDWRITING ANALYSIS
# =====================================================
st.header("✍ Handwriting Detection")

hand_file = st.file_uploader(
    "Upload Handwriting Image",
    type=["jpg", "jpeg", "png"]
)

if hand_file is not None:

    try:
        image = Image.open(hand_file)
        st.image(image)

        img = np.array(image)
        img = cv2.resize(img, (128,128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        if handwriting_model is not None:

            prediction = handwriting_model.predict(img)

            if prediction[0][0] > 0.5:
                st.error("⚠ Parkinson symptoms detected in handwriting")
            else:
                st.success("✅ Handwriting looks normal")

    except Exception as e:
        st.error(f"Handwriting processing error: {e}")

st.markdown("---")
st.write("AI Parkinson Detection System | Multi-Modal ML")
