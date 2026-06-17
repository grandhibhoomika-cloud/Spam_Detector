import os
import streamlit as st
import joblib

# Page config
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

# Header UI
st.markdown("<h1 style='text-align:center; color:green;'>📩 Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("### Check if your message is Spam or Not")

# File paths (IMPORTANT)
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

# Load model (SAFE METHOD)
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Input UI
msg = st.text_area("✍️ Enter your message", height=150)

# Predict button
if st.button("🚀 Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message")
    else:
        msg_vec = vectorizer.transform([msg])
        pred = model.predict(msg_vec)[0]

        st.markdown("---")

        if pred == "spam":
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is NOT SPAM")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit + ML")
