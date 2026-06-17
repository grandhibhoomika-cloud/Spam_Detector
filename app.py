import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title("📩 Spam Email Detector")

# Load dataset
df = pd.read_csv("spam.csv")

X = df["message"]
y = df["label"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Input
msg = st.text_area("Enter message")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Enter a message")
    else:
        msg_vec = vectorizer.transform([msg])
        prediction = model.predict(msg_vec)[0]

        if prediction == "spam":
            st.error("🚨 SPAM")
        else:
            st.success("✅ NOT SPAM")
