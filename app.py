import streamlit as st
import pickle
import string
import nltk
import os

# ===== NLTK SETUP - THIS FIXES THE ERROR =====
# Set the NLTK data directory
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data with verification
required_nltk = ['punkt', 'stopwords']
for package in required_nltk:
    try:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
    except LookupError:
        nltk.download(package, download_dir=nltk_data_dir)

# ===== ORIGINAL CODE WITH MINIMAL CHANGES =====
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))  # Preload for better performance

def txt_trans(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in STOPWORDS and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)

# Load models
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")
    st.stop()

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")
if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message!")
    else:
        try:
            #1. preprocess
            transformed_sms = txt_trans(input_sms)
            #2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            #3. predict
            result = model.predict(vector_input)
            #4. display
            st.header("SPAM 🚨" if result == 1 else "NOT SPAM ✅")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
