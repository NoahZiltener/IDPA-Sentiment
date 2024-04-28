import streamlit as st
from googleapiclient.discovery import build
from youtube import get_video_comments
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from googletrans import Translator
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')

API_KEY = 'AIzaSyBPxoC3inmp5KbBZ5bYw6c2GIsTWmUFpBM'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
sia = SentimentIntensityAnalyzer()
translator = Translator()

st.set_page_config(page_title="Youtube Sentiment Analyse", page_icon="📈")

st.title('Youtube Sentiment Analyse')

st.text('Geben Sie die URL des Youtube-Videos ein, das Sie analysieren möchten.')

video_url = st.text_input('Youtube Video URL', '')

translate = st.toggle('Kommentare Übersetzen', False)

results = list()

if video_url:
    with st.spinner('Kommentare werden abgerufen...'):
        video_id = video_url.split('=')[-1]
        comments = get_video_comments(youtube, part='snippet', videoId=video_id, textFormat='plainText')
        for comment in comments:
            if translate:
                comment['textDisplay'] = translator.translate(comment['textDisplay'], dest='en').text
            sentiment = sia.polarity_scores(comment['textDisplay'])
            results.append({'comment': comment['textDisplay'], 'orginalComment': comment['textOriginal'], 'sentiment': sentiment, 'last_updated': comment['updatedAt']})
    st.success('Kommentare erfolgreich abgerufen!')
    df = pd.DataFrame(results)

    df['neg'] = df['sentiment'].apply(lambda x: x['neg']).astype(float)
    df['neu'] = df['sentiment'].apply(lambda x: x['neu']).astype(float)
    df['pos'] = df['sentiment'].apply(lambda x: x['pos']).astype(float)
    df['compound'] = df['sentiment'].apply(lambda x: x['compound']).astype(float)

    df['last_updated'] = pd.to_datetime(df['last_updated'])

    st.title('Kommentar Sentiment Analyse')

    exclude_neutral_1 = st.sidebar.checkbox('Kommentare mit neutralem Wert von 1 ausschließen', True)

    if exclude_neutral_1:
        df_filtered = df[df['neu'] != 1]
    else:
        df_filtered = df

    st.header('Anzahl der Kommentare')
    st.write(f"Total: {len(df_filtered)}")

    st.header('Durchschnittliche Sentiments')
    st.write(df_filtered[['neg', 'neu', 'pos', 'compound']].mean())

    def sentiment_to_emoji(value):
        if value > 0.05:
            return '😀'
        elif value < -0.05:
            return '😞'
        else:
            return '😐'
    st.write(sentiment_to_emoji(df_filtered['compound'].mean()))

    st.header('Top-Sentiments')
    top_positive = df_filtered.loc[df_filtered['compound'].idxmax()]
    top_negative = df_filtered.loc[df_filtered['compound'].idxmin()]

    st.subheader('Positivster Kommentar')
    st.write(top_positive['orginalComment'])
    st.write(top_positive['compound'])

    st.subheader('Negativster Kommentar')
    st.write(top_negative['orginalComment'])
    st.write(top_negative['compound'])

    # Durchschnittlicher Sentiment pro Tag
    st.header('Durchschnittlicher Sentiment pro Tag')

    # Gruppieren der Daten nach Datum und Berechnen des durchschnittlichen 'compound' Wertes
    average_sentiment_per_day = df_filtered.resample('D', on='last_updated')['compound'].mean()

    st.line_chart(average_sentiment_per_day)


