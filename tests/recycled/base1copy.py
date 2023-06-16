import sys
sys.path.append("..")
import streamlit as st
import pandas as pd
import snscrape.modules.twitter as sntwitter
import plotly.express as px
import re
import nltk
import base64
from src.data.make_dataset import get_base64_of_bin_file, set_png_as_page_bg

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

st.title('Social Listening App')
rad = st.sidebar.radio("Navigation", ['Home', 'Twitter', 'Facebook', 'Instagram', 'LinkedIn', 'YouTube', 'ExploreAI Webpage'])
if rad == 'Twitter':

    set_png_as_page_bg('imgs/twitter.jpg')    

    query = st.text_input('Enter query here', value='@explore_ai_acad')

    # Define the cache timeout
    CACHE_TIMEOUT = 300  # seconds (5 minutes)

    # Define the function to scrape tweets and cache the results
    @st.cache_data(ttl=CACHE_TIMEOUT)
    def scrape_tweets(query):
        tweets = []
        lim = '2013'

        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            if str(tweet.date.year) == lim:
                break
            else:
                tweets.append([tweet.date, tweet.user.username, tweet.user.location, 
                        tweet.rawContent, tweet.url, tweet.renderedContent, 
                        tweet.id, tweet.user.displayname, 
                        tweet.user.followersCount, tweet.user.friendsCount, 
                        tweet.user.statusesCount, tweet.user.favouritesCount, 
                        tweet.replyCount, tweet.retweetCount, tweet.likeCount, 
                        tweet.quoteCount, tweet.lang, tweet.retweetedTweet, 
                        tweet.quotedTweet])
        

        df = pd.DataFrame(tweets, columns=['Date', 'Username', 'Location', 'Tweet', 
                                    'URL', 'RenderedContent', 'id', 
                                    'DisplayName', 'FollowersCount', 
                                    'FriendsCount', 'StatusesCount', 
                                    'FavouritesCount', 'ReplyCount', 
                                    'RetweetCount', 'LikeCount', 'QuoteCount', 
                                    'Language', 'RetweetedTweet', 'QuotedTweet'])
        return df
    

    # Scrape tweets based on the search query
    df = scrape_tweets(query)

    # Display the resulting dataframe using Streamlit's table component

    st.dataframe(df)

    # Tweet to show the cleaned tweet
    
    df_train = pd.read_csv('tweets_about_ExploreAI_with_location.csv')
    fig = px.histogram(df_train, x='Sentiment', color='Sentiment', title='Sentiments of Training Data')
    st.plotly_chart(fig, use_container_width=True)

    @st.cache_data
    def hashtag_extract(tweet):
 
    
        hashtags = []
        
        for i in tweet:
            ht = re.findall(r"#(\w+)", i)
            hashtags.append(ht)
            
        hashtags = sum(hashtags, [])
        frequency = nltk.FreqDist(hashtags)
        
        hashtag_df = pd.DataFrame({'hashtag': list(frequency.keys()),
                        'count': list(frequency.values())})
        hashtag_df = hashtag_df.nlargest(15, columns="count")

        return hashtag_df
    all = hashtag_extract(df['Tweet'].str.lower())
    fig1 = px.bar(all, x='count', y='hashtag', color='hashtag', title='Top 20 Hashtags of Tweets')
    st.plotly_chart(fig1, use_container_width=True)

if rad == 'Facebook':
    set_png_as_page_bg('imgs/youtube.jpg')
    st.file_uploader('Upload File')
    
if rad == 'Instagram':
    st.file_uploader('Upload File')

if rad == 'LinkedIn':
    st.file_uploader('Upload File')

if rad == 'ExploreAI Webpage':
    st.file_uploader('Upload File')      