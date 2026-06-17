import os
import pickle
import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered"
)

# =========================
# DEBUG SECTION (REMOVE LATER IF YOU WANT CLEAN UI)
# =========================
st.write("📁 Current Working Directory:")
st.write(os.getcwd())

st.write("📂 Files in deployment folder:")
st.write(os.listdir("."))

# =========================
# HEADER UI
# =========================
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📩 Spam Detector</h1>",
    unsafe_allow_html=True
)

st.markdown("### Check whether your message is Spam or Not")

# =========================
# LOAD MODEL SAFELY
# =========================
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

st.write("🔍 Model Exists:", os.path.exists(model_path))
st.write("🔍 Vectorizer Exists:", os.path.exists(vectorizer_path))

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# =========================
# INPUT UI
# =========================
msg = st.text_area("✍️ Enter your message:", height=150)

# =========================
# PREDICTION BUTTON
# =========================
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

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit + Machine Learning")
