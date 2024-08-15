import tensorflow as tf
from tensorflow.keras.datasets import imdb
import streamlit as st
from tensorflow.keras.utils import pad_sequences

model = tf.keras.models.load_model("imdb_model.keras")

word_index = imdb.get_word_index()

def preprocess_review(review):
    words = review.lower().split()
    words = [word_index[word]for word in words]
    padded_review = pad_sequences([words], maxlen=500)
    return padded_review

def predict(review):
    proce_review = preprocess_review(review)
    preds = model.predict(proce_review)
    if preds[0][0] >= 0.5:
        return f'Predicted Positive with "{preds[0][0]}" accuracy.'
    else:
        return f"Predicted Negative with '{preds[0][0]}' accuracy."
    

st.title("Word Embedding model for imdb movie review")
st.title('Analysis')
review = st.text_area("Enter your review: ")

if st.button('Classify'):
    st.title(predict(review))
else:
    st.write('Please enter a review')