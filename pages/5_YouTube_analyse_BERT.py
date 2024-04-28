import streamlit as st
from googleapiclient.discovery import build
from youtube import get_video_comments
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from googletrans import Translator
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

API_KEY = '<API_KEY>'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    return_all_scores=True
)

translator = Translator()

nltk.download('vader_lexicon')

st.set_page_config(page_title="YouTube Kommentar Sentiment Analyse BERT", page_icon="📈")

st.title('YouTube Kommentar Sentiment Analyse BERT')

st.text('Geben Sie die URL des Youtube-Videos ein, das Sie analysieren möchten.')

video_url = st.text_input('Youtube Video URL', '')

results = list()

if video_url:
    with st.spinner('Text wird analysiert...'):
        video_id = video_url.split('=')[-1]
        comments = get_video_comments(youtube, part='snippet', videoId=video_id, textFormat='plainText')

        for comment in comments:
            if len(comment['textDisplay']) > 500:
                continue
            sentiment = classifier(comment['textDisplay'])
            results.append({'comment': comment['textDisplay'], 'orginalComment': comment['textOriginal'], 'sentiment': sentiment, 'last_updated': comment['updatedAt']})
            print('Kommentar analysiert')
        st.success('Kommentare erfolgreich abgerufen!')
        
        df = pd.DataFrame(results)

        st.title('Gesammtbewertung')

        # Initialize counters for each sentiment
        total_positive = 0
        total_neutral = 0
        total_negative = 0

        # Iterate over the results
        for result in df['sentiment']:
            # Sum up the scores for each sentiment
            for s in result[0]:
                if s['label'] == 'positive':
                    total_positive += s['score']
                elif s['label'] == 'neutral':
                    total_neutral += s['score']
                elif s['label'] == 'negative':
                    total_negative += s['score']

        # Determine the prevailing emotion
        if total_positive > total_neutral and total_positive > total_negative:
            prevailing_emotion = 'positive'
        elif total_neutral > total_positive and total_neutral > total_negative:
            prevailing_emotion = 'neutral'
        else:
            prevailing_emotion = 'negative'

        st.subheader('Emotion')
        st.markdown(f'Die Kommentare sind **{prevailing_emotion}**.')

        import matplotlib.pyplot as plt

        # Create a new column for the emotion with the highest score
        df['prevailing_emotion'] = df['sentiment'].apply(lambda x: max(x[0], key=lambda s: s['score'])['label'])

        # Count the number of comments for each emotion
        emotion_counts = df['prevailing_emotion'].value_counts()

        # Define a function to format the autopct parameter
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return '{p:.1f}%  ({v:d})'.format(p=pct,v=val)
            return my_autopct
        
        # Create a pie chart
        plt.figure(figsize=(10, 6))
        plt.pie(emotion_counts, labels=emotion_counts.index, autopct=make_autopct(emotion_counts), startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Anzahl der Kommentare pro Emotion')
        st.pyplot(plt)

        # Convert the date column to datetime and extract the date
        df['date'] = pd.to_datetime(df['last_updated']).dt.date

        # Create a new DataFrame that counts the number of comments for each emotion per day
        emotion_counts_per_day = df.groupby(['date', 'prevailing_emotion']).size().unstack(fill_value=0)

        # Plot the data
        emotion_counts_per_day.plot(kind='line', figsize=(10, 6))
        plt.title('Anzahl der Kommentare pro Emotion über die Zeit')
        plt.xlabel('Datum')
        plt.ylabel('Anzahl der Kommentare')
        st.pyplot(plt)

        st.write(df)
        st.table(df)