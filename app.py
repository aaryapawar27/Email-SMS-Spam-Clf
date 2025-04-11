import streamlit as st
import nltk
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK resources are downloaded before use
@st.cache_resource
def setup_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    return True

setup_nltk()

ps = PorterStemmer()

def txt_trans(text):
    # Lowercase
    text = text.lower()
    # Tokenize
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    try:
        # Preprocess
        transformed_sms = txt_trans(input_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Display
        if result == 1:
            st.header("SPAM")
        else:
            st.header("NOT SPAM")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
