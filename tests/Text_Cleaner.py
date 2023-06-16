import re
import nltk
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')




def hashtag_extract(text, num_hashtags):
    
    """
    This function takes in text with hashtags and extracts the top 10 hashtag(s) using regular expressions
    These hashtags are stored in a seperate dataframe 
    along with a count of how frequently they occur
    
    Input:
    text: original statements
           datatype: 'str'
    num_hashtags: top number of hashtags with the highest count.
                    datatype: 'int'
           
    Output:
    hashtag_df: dataframe containing the top hashtags in the texts
              datatype: dataframe         
    """
    
    hashtags = []
    
    for i in text:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
        
    hashtags = sum(hashtags, [])
    frequency = nltk.FreqDist(hashtags)
    
    hashtag_df = pd.DataFrame({'hashtag': list(frequency.keys()),
                       'count': list(frequency.values())})
    hashtag_df = hashtag_df.nlargest(num_hashtags, columns="count")

    return hashtag_df




def clean_text(text):
    text = text.str.lower()
    text = text.to_list()
    text = [remove_stopwords(i) for i in text]
    text = [re.sub(r'#(\w+)', '', i) for i in text]
    text = [re.sub(r'd\+', '', i) for i in text]
    text = [re.sub(r'[^\w\s]', '', i) for i in text]
    text = [re.sub(r'http\S+', '', i, flags=re.MULTILINE) for i in text]
    #text = [nltk.word_tokenize(i) for i in text]
    
    
    return text


def lemma(df, col):
    
    """
    This function modifies the original train dataframe.
    A new column for the length of each tweet is added.
    The tweets are then tokenized and each word is assigned a part of speech tag 
    before being lemmatized
    
    Input:
    df: original dataframe
        datatype: dataframe 
        
    Output:
    df: modified dataframe
        datatype: dataframe
    """
    
    df['text_length'] = df[col].str.len()
    df['tokenized'] = df[col].apply(word_tokenize)
    df['pos_tags'] = df['tokenized'].apply(nltk.tag.pos_tag)

    def get_wordnet_pos(tag):

        if tag.startswith('J'):
            return wordnet.ADJ

        elif tag.startswith('V'):
            return wordnet.VERB

        elif tag.startswith('N'):
            return wordnet.NOUN

        elif tag.startswith('R'):
            return wordnet.ADV
    
        else:
            return wordnet.NOUN
        
    wnl = WordNetLemmatizer()
    df['pos_tags'] = df['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    df['lemmatized'] = df['pos_tags'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
    df['lemmatized'] = [' '.join(map(str, l)) for l in df['lemmatized']]
      
    return df




def frequency(text):
    
    """
    This function determines the frequency of each word in a collection of tweets 
    and stores the 25 most frequent words in a dataframe, 
    sorted from most to least frequent
    
    Input: 
    tweet: original tweets
           datatype: 'str'
           
    Output: 
    frequency: dataframe containing the top 25 words 
               datatype: dataframe          
    """
    
    # Count vectorizer excluding english stopwords
    cv = CountVectorizer(stop_words='english')
    words = cv.fit_transform(text)
    
    # Count the words in the tweets and determine the frequency of each word
    sum_words = words.sum(axis=0)
    words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
    # Create a dataframe to store the top 25 words and their frequencies
    frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])
    frequency = frequency.head(25)
    
    return frequency