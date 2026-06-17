import streamlit as st
import pickle

# Page config
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title section
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📩 Spam Email Detector</h1>", unsafe_allow_html=True)

st.write("")

# Input box inside a styled container
st.markdown("### Enter your message below:")

msg = st.text_area("", height=150)

# Button
if st.button("Check Message"):
    if msg.strip() == "":
        st.warning("Please enter a message")
    else:
        data = vectorizer.transform([msg])
        result = model.predict(data)[0]

        st.write("")

        # Output styling
        if result == "spam":
            st.markdown(
                "<h2 style='text-align: center; color: red;'>🚨 This is SPAM</h2>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h2 style='text-align: center; color: green;'>✅ This is NOT SPAM</h2>",
                unsafe_allow_html=True
            )

# Footer
st.write("")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built using Machine Learning + Streamlit</p>",
    unsafe_allow_html=True
)