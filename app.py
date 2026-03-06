import streamlit as st
import numpy as np
import pandas as pd
import cv2
import librosa
import tempfile
from datetime import datetime
from tensorflow.keras.models import load_model
from audiorecorder import audiorecorder

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="AI Parkinson Detection", layout="wide")

st.title("🧠 AI-Based Parkinson’s Multi-Modal Detection System")
st.write("Speech + Spiral + Handwriting Analysis")

# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------
@st.cache_resource
def load_models():
    speech = load_model("parkinson_cnn_model.h5")
    spiral = load_model("spiral_model.h5")
    handwriting = load_model("handwriting_model.keras")
    return speech, spiral, handwriting

speech_model, spiral_model, handwriting_model = load_models()

# ------------------------------------------------
# SPEECH PROCESSING
# ------------------------------------------------
def get_speech_prob(audio_path):

    y, sr = librosa.load(audio_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
    mfcc = cv2.resize(mfcc, (128, 128))
    mfcc = mfcc / np.max(np.abs(mfcc))
    mfcc = mfcc.reshape(1, 128, 128, 1)

    prediction = speech_model.predict(mfcc)
    return float(prediction[0][1])  # Parkinson class index

# ------------------------------------------------
# SPIRAL PROCESSING
# ------------------------------------------------
def get_spiral_prob(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = spiral_model.predict(img)
    return float(prediction[0][0])

# ------------------------------------------------
# HANDWRITING PROCESSING
# ------------------------------------------------
def get_handwriting_prob(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = handwriting_model.predict(img)
    return float(prediction[0][0])

# ------------------------------------------------
# SESSION STATE
# ------------------------------------------------
if "speech_prob" not in st.session_state:
    st.session_state.speech_prob = None

# ==================================================
# 🎙 LIVE SPEECH
# ==================================================
st.subheader("🎙 Live Speech Recording")

audio = audiorecorder("Click to record", "Recording... Click to stop")

if len(audio) > 0:
    st.audio(audio.export().read())

    if st.button("Analyze Live Speech"):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(temp_file.name, format="wav")
        prob = get_speech_prob(temp_file.name)
        st.session_state.speech_prob = prob
        st.success(f"Live Speech Probability: {prob:.2f}")

# ==================================================
# 📂 UPLOAD SPEECH
# ==================================================
st.subheader("📂 Upload Speech File (.wav)")

uploaded_audio = st.file_uploader("Upload Speech Audio", type=["wav"])

if uploaded_audio is not None:
    st.audio(uploaded_audio)

    if st.button("Analyze Uploaded Speech"):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(uploaded_audio.read())
        prob = get_speech_prob(temp_file.name)
        st.session_state.speech_prob = prob
        st.success(f"Uploaded Speech Probability: {prob:.2f}")

# ==================================================
# 🌀 SPIRAL
# ==================================================
st.subheader("🌀 Upload Spiral Drawing")

spiral_file = st.file_uploader("Upload Spiral Image", type=["png", "jpg", "jpeg"])

# ==================================================
# ✍ HANDWRITING
# ==================================================
st.subheader("✍ Upload Handwriting Sample")

handwriting_file = st.file_uploader(
    "Upload Handwriting Image",
    type=["png", "jpg", "jpeg"],
    key="handwriting"
)

# ==================================================
# 🤖 RUN ANALYSIS
# ==================================================
if st.button("Run Full AI Analysis"):

    if st.session_state.speech_prob is None:
        st.warning("Please analyze speech first.")
        st.stop()

    if spiral_file is None:
        st.warning("Please upload spiral image.")
        st.stop()

    speech_prob = st.session_state.speech_prob
    spiral_prob = get_spiral_prob(spiral_file)

    handwriting_prob = 0
    if handwriting_file is not None:
        handwriting_prob = get_handwriting_prob(handwriting_file)

    # Weighted Fusion (handwriting strongest model)
    fusion_prob = (
        0.3 * speech_prob +
        0.3 * spiral_prob +
        0.4 * handwriting_prob
    )

    # Risk Level
    if fusion_prob < 0.4:
        risk_level = "Low Risk"
    elif fusion_prob < 0.7:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"

    # Display Results
    st.subheader("📊 AI Analysis Results")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Speech", f"{speech_prob:.2f}")
    col2.metric("Spiral", f"{spiral_prob:.2f}")
    col3.metric("Handwriting", f"{handwriting_prob:.2f}")
    col4.metric("Final Risk", f"{fusion_prob:.2f}")

    st.progress(fusion_prob)

    if risk_level == "Low Risk":
        st.success("✅ Low Risk")
    elif risk_level == "Moderate Risk":
        st.warning("⚠ Moderate Risk")
    else:
        st.error("🚨 High Risk - Consult Neurologist")

    st.info("⚠ This AI tool is for screening purposes only.")

    # Save History
    new_entry = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Speech_Prob": speech_prob,
        "Spiral_Prob": spiral_prob,
        "Handwriting_Prob": handwriting_prob,
        "Fusion_Prob": fusion_prob,
        "Risk_Level": risk_level
    }

    try:
        df = pd.read_csv("patient_history.csv")
    except:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv("patient_history.csv", index=False)

    st.subheader("📁 Patient History")
    st.dataframe(df)