import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from googletrans import Translator

sia = SentimentIntensityAnalyzer()
translator = Translator()

st.set_page_config(page_title="Text Sentiment Analyse VADER", page_icon="📈")

st.title('Text Sentiment Analyse VADER')

st.text('Geben Sie eine Text ein, den Sie analysieren möchten.')

text = st.text_input('Text', '')

translate = st.toggle('Übersetzen', False)

if text:
    with st.spinner('Text wird analysiert...'):
        if translate:
            text = translator.translate(text, dest='en').text
        sentiment = sia.polarity_scores(text)
        st.success('Text erfolgreich analysiert!')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Positive', sentiment['pos'])
        col2.metric('Neutral', sentiment['neu'])
        col3.metric('Negative', sentiment['neg'])
        col4.metric('Compound', sentiment['compound'])