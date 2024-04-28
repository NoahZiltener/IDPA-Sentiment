import streamlit as st
from googleapiclient.discovery import build
from youtube import get_video_comments
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from googletrans import Translator
import pandas as pd
import matplotlib.pyplot as plt

API_KEY = '<API_KEY>'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
sia = SentimentIntensityAnalyzer()
translator = Translator()

nltk.download('vader_lexicon')

st.set_page_config(page_title="YouTube Kommentar Sentiment Analyse VADER", page_icon="📈")

st.title('YouTube Kommentar Sentiment Analyse VADER')

st.text('Geben Sie die URL des Youtube-Videos ein, das Sie analysieren möchten.')

video_url = st.text_input('Youtube Video URL', '')
translate = st.toggle('Kommentare Übersetzen', False)

results = list()

if video_url:
    with st.spinner('Text wird analysiert...'):
        video_id = video_url.split('=')[-1]
        comments = get_video_comments(youtube, part='snippet', videoId=video_id, textFormat='plainText')
        for comment in comments:
            if translate:
                comment['textDisplay'] = translator.translate(comment['textDisplay'], dest='en').text
            sentiment = sia.polarity_scores(comment['textDisplay'])
            results.append({'comment': comment['textDisplay'], 'orginalComment': comment['textOriginal'], 'sentiment': sentiment, 'last_updated': comment['updatedAt']})
        st.success('Kommentare erfolgreich abgerufen!')
        
        df = pd.DataFrame(results)

        # Filter out comments longer than 500 characters
        df = df[df['comment'].str.len() <= 500]

        st.title('Gesammtbewertung')
        # Extract compound scores into a separate Series
        compound_scores = df['sentiment'].apply(lambda x: x['compound'])

        # Convert compound scores to numeric
        compound_scores = pd.to_numeric(compound_scores, errors='coerce')

        # Calculate mean and median
        mean_compound = compound_scores.mean()
        median_compound = compound_scores.median()

        st.subheader('Emotion')
        if mean_compound >= 0.05:
            st.markdown('Die Kommentare sind **positiv**.')
        elif mean_compound <= -0.05:
            st.markdown('Die Kommentare sind **negativ**.')
        else:
            st.markdown('Die Kommentare sind **neutral**.')

        st.markdown(f'Der Durchschnitt der Compound-Werte ist: **{mean_compound}**')
        st.markdown(f'Der Median der Compound-Werte ist: **{median_compound}**')

        import matplotlib.pyplot as plt

        # Classify comments into positive, negative and neutral
        df['sentiment_category'] = df['sentiment'].apply(lambda x: 'positiv' if x['compound'] >= 0.05 else ('negativ' if x['compound'] <= -0.05 else 'neutral'))

        # Count the number of each category
        sentiment_counts = df['sentiment_category'].value_counts()

        # Define a function to format the autopct parameter
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return '{p:.1f}%  ({v:d})'.format(p=pct,v=val)
            return my_autopct

        # Create pie chart
        plt.figure(figsize=(10, 7))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct=make_autopct(sentiment_counts), startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.pyplot(plt)

        # Find the index of the comment with the highest compound score
        max_compound_idx = compound_scores.idxmax()

        # Find the index of the comment with the lowest compound score
        min_compound_idx = compound_scores.idxmin()

        # Extract the comments with the highest and lowest compound scores
        most_positive_comment = df.loc[max_compound_idx, 'orginalComment']
        most_negative_comment = df.loc[min_compound_idx, 'orginalComment']

        # Display the most positive and most negative comments
        st.subheader('Most Positive Comment')
        st.markdown(f'**Comment:** {most_positive_comment}')
        st.markdown(f'**Compound Score:** {compound_scores[max_compound_idx]}')

        st.subheader('Most Negative Comment')
        st.markdown(f'**Comment:** {most_negative_comment}')
        st.markdown(f'**Compound Score:** {compound_scores[min_compound_idx]}')

        st.header('Average Compound Score per Hour')

        # Convert the date column to datetime
        df['date'] = pd.to_datetime(df['last_updated'])
        df['compound'] = df['sentiment'].apply(lambda x: x['compound'])

        # Group by hour and calculate the mean compound score
        average_hourly_scores = df.groupby(df['date'].dt.floor('H'))['compound'].mean()

        # Create a plot of the average hourly scores
        plt.figure(figsize=(10, 7))
        plt.plot(average_hourly_scores.index, average_hourly_scores.values)
        plt.xlabel('Hour')
        plt.ylabel('Average Compound Score')
        plt.title('Average Compound Score per Hour')

        st.pyplot(plt)

        st.header('Average Compound Score per Day')
        # Convert the date column to datetime
        df['date'] = pd.to_datetime(df['last_updated'])
        df['compound'] = df['sentiment'].apply(lambda x: x['compound'])

        # Group by day and calculate the mean compound score
        average_daily_scores = df.groupby(df['date'].dt.date)['compound'].mean()

        # Create a plot of the average daily scores
        plt.figure(figsize=(10, 7))
        plt.plot(average_daily_scores.index, average_daily_scores.values)
        plt.xlabel('Date')
        plt.ylabel('Average Compound Score')
        plt.title('Average Compound Score per Day')

        st.pyplot(plt)

        st.write(df)