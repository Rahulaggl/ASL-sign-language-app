import streamlit as st
from model_utils import load_trained_model, predict_sequence
from data_utils import preprocess_uploaded_images
import pyttsx3  # 🗣️ NEW

# 🔊 Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

st.set_page_config(page_title="ASL Sign Recognition", layout="centered")
st.title("🧠 ASL Sign Recognition")
st.write("Upload a sequence of 10 hand gesture images (JPEG) to classify the ASL sign.")

uploaded_files = st.file_uploader("Upload 10 JPEG images", type=["jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing..."):
        sequence, error = preprocess_uploaded_images(uploaded_files)
        if error:
            st.error(error)
        else:
            model = load_trained_model()
            prediction_probs, predicted_class = predict_sequence(model, sequence)

            st.success(f"✋ Predicted Sign: **{predicted_class}**")
            st.bar_chart(prediction_probs[0])

            # 🔈 Speak out the result
            engine.say(f"The predicted sign is {predicted_class}")
            engine.runAndWait()
else:
    st.info("📥 Please upload exactly 10 images to start prediction.")
