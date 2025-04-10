import streamlit as st
import pickle
import string
import nltk
import os

# ===== NLTK SETUP - PROPER CONFIGURATION =====
# Set up NLTK data directory
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download ONLY the required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)

# ===== MAIN APP CODE =====
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))  # Convert to set for faster lookups

def txt_trans(text):
    # Lowercase
    text = text.lower()
    # Tokenize (uses punkt internally)
    tokens = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric
    tokens = [token for token in tokens if token.isalnum()]
    
    # Remove stopwords and punctuation
    tokens = [token for token in tokens 
              if token not in STOPWORDS 
              and token not in string.punctuation]
    
    # Stemming
    tokens = [ps.stem(token) for token in tokens]
    
    return ' '.join(tokens)

# Load models with error handling
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
            transformed_sms = txt_trans(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            st.header("SPAM 🚨" if result == 1 else "NOT SPAM ✅")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
