import streamlit as st
from googletrans import Translator
from textblob import TextBlob

translator = Translator()

st.set_page_config(page_title="Text Sentiment Analyse TextBlob", page_icon="📈")

st.title('Text Sentiment Analyse TextBlob')

st.text('Geben Sie eine Text ein, den Sie analysieren möchten.')

text = st.text_input('Text', '')

translate = st.toggle('Übersetzen', False)

if text:
    with st.spinner('Text wird analysiert...'):
        if translate:
            text = translator.translate(text, dest='en').text
        blob = TextBlob(text)
        sentiment = blob.sentiment
        st.success('Text erfolgreich analysiert!')
        st.write(sentiment)