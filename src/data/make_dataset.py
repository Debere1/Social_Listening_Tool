"""Code to process data."""

from typing import List
import random
import base64
import streamlit as st
import snscrape.modules.twitter as sntwitter
import pandas as pd
from textblob import TextBlob
import string
import nltk
nltk.download('stopwords')


def generate_data(
    length: int,
    lowest: int = 0,
    highest: int = 100,
) -> List:
    """Generate data.

    A simple example to show how to separate out your code base in a clean way.

    See notebooks/0.0-example.ipynb

    Args:
        length (int): How many elements to have in the generated list.
        lowest (int, optional): Lowest possible int in the randomly generated list. Defaults to 0.
        highest (int, optional): Highest possible int in the randomly generated list. Defaults to 100.

    Returns:
        List: A list of random numbers.
    """
    return [random.randint(lowest, highest) for _ in range(length)]

#code to import background picture
def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
        bin_str = get_base64_of_bin_file(png_file)
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        </style>
        ''' % bin_str
    
        st.markdown(page_bg_img, unsafe_allow_html=True)
        return

# Define the list of stopwords
stopwords = nltk.corpus.stopwords.words('english')

# Define the function to clean tweets
def clean_tweet(tweet):
    # Remove punctuation
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    tweet = tweet.lower()
    # Spelling correction
    tweet = tweet.apply(lambda x: str(TextBlob(x).correct()))
    # Tokenize the tweet using TextBlob
    tokens = TextBlob(tweet).words
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords]
    # Join tokens to form a cleaned tweet
    cleaned_tweet = ' '.join(tokens)
    return cleaned_tweet