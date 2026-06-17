import streamlit as st
import pickle

# Page config
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

# Header UI
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📩 Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("### Check whether a message is Spam or Not")

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Input box UI
msg = st.text_area("✍️ Enter your message below:", height=150)

# Button
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
st.markdown("Built with ❤️ using Streamlit + Machine Learning")
