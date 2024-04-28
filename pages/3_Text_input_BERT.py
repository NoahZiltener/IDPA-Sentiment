import streamlit as st
from googletrans import Translator
from transformers import pipeline

classifier = pipeline('sentiment-analysis')
translator = Translator()

st.set_page_config(page_title="Text Sentiment Analyse Bert", page_icon="📈")

st.title('Text Sentiment Analyse BERT')

st.text('Geben Sie eine Text ein, den Sie analysieren möchten.')

text = st.text_input('Text', '')

translate = st.toggle('Übersetzen', False)

if text:
    with st.spinner('Text wird analysiert...'):
        if translate:
            text = translator.translate(text, dest='en').text
        result = classifier(text)[0]
        st.success('Text erfolgreich analysiert!')
        st.write(result)