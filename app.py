import streamlit as st
import pickle
import string
import nltk
import os

# ===== 1. NLTK SETUP - THIS WILL FIX THE ERROR =====
# Create and configure NLTK data directory
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download and verify ONLY the necessary NLTK data
REQUIRED_NLTK_DATA = ['punkt', 'stopwords']

for package in REQUIRED_NLTK_DATA:
    try:
        # Check if package is already available
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
    except LookupError:
        # If not found, download it
        nltk.download(package, download_dir=nltk_data_path)
        # Verify download was successful
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError:
            st.error(f"Failed to download NLTK {package} data")
            st.stop()

# ===== 2. TEXT PROCESSING =====
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Preload resources for better performance
ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))
PUNCTUATIONS = set(string.punctuation)

def txt_trans(text):
    try:
        # Lowercase
        text = text.lower()
        # Tokenize (uses punkt internally)
        tokens = nltk.word_tokenize(text)
        # Remove non-alphanumeric
        tokens = [token for token in tokens if token.isalnum()]
        # Remove stopwords and punctuation
        tokens = [token for token in tokens 
                 if token not in STOPWORDS 
                 and token not in PUNCTUATIONS]
        # Stemming
        tokens = [ps.stem(token) for token in tokens]
        return ' '.join(tokens)
    except Exception as e:
        st.error(f"Text processing error: {str(e)}")
        return ""

# ===== 3. MODEL LOADING =====
@st.cache_resource
def load_models():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

tfidf, model = load_models()

# ===== 4. STREAMLIT UI =====
st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")
if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message!")
    elif model is None:
        st.error("Model not loaded - check server logs")
    else:
        transformed = txt_trans(input_sms)
        if transformed:  # Only proceed if preprocessing succeeded
            vector = tfidf.transform([transformed])
            result = model.predict(vector)[0]
            st.header("SPAM 🚨" if result == 1 else "NOT SPAM ✅")
