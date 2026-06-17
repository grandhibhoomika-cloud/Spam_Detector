import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Page config
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

# Header UI
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📩 Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("### Check if your message is Spam or Not Spam")

# Load dataset
df = pd.read_csv("spam.csv")

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Input UI
msg = st.text_area("✍️ Enter your message here:")

# Button
if st.button("🚀 Check Message"):
    if msg.strip() == "":
        st.warning("Please enter a message")
    else:
        msg_vec = vectorizer.transform([msg])
        prediction = model.predict(msg_vec)[0]

        if prediction.lower() == "spam":
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is NOT SPAM")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit + ML")
