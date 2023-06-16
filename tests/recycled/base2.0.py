import sys
sys.path.append("..")
import streamlit as st
import pandas as pd
import snscrape.modules.twitter as sntwitter
import plotly.express as px
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

import base64
from src.data.make_dataset import get_base64_of_bin_file, set_png_as_page_bg

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')



# Page setting
st.set_page_config(layout="wide")
import Filters   
## Home Page
    
rad = st.sidebar.radio("Navigation", ['Home', 'Twitter','Facebook', 'Instagram', 'LinkedIn'])


if rad == 'Home':   
   st.title('Social Listening App')
    
 ## Twitter Page
   
if rad == 'Twitter':
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
   

    set_png_as_page_bg('imgs/T2.gif') 
    
    # loading the dataset
    #df1= pd.read_csv('C:/Users/Tiro/OneDrive/Documents/GitHub/internship-project-2207-14/Sentimental_Tweets.csv')
    df1 = pd.read_csv('C:/Users/Tiro/OneDrive/Documents/GitHub/internship-project-2207-14/ExploreAI social media data - Twitter Sprout.csv')
    
    #cleaning the Dataset with the imported NLP Funtion
    #Twitter_NLP.cleaning(df)
    
    #Filtering the cleaned dataset  with the imported Filtering Function
    #Filters.filter_dataframe(df1)
    df = Filters.filter_dataframe(df1)
    
    # Row a
    likes= df["Likes"].sum()
    comments= df["Comments"].sum()
    engagements= df["Engagements"].sum()
    #imps= df["Impressions"].sum()
    #reach= df["Potential Reach"].sum()
    
    
    col1, col2, col3, = st.columns(3)
    col1.metric(label=" ### Comments", value=int(comments), delta="")
    col2.metric(label=" ### Engagements", value=int(engagements), delta="")
    col3.metric(label=" ### Likes", value=int(likes), delta="")
    #col4.metric(label=" ### Impressions", value=int(imps), delta="")
    #col5.metric(label=" ### Potential Reach", value=int(reach), delta="")
    
   
    
    #Row B
    c1, c2 = st.columns((7,3))
    with c1:
        st.markdown('### Line graph')
        grp= df[['Date','Impressions', 'Likes', 'Engagements', 'Comments','Potential Reach']].groupby(['Date']).sum().reset_index()
        st.write(px.line(df, x= 'Date', y= 'Impressions',color_discrete_sequence=px.colors.qualitative.Plotly_r))
     
        
    with c2:
        st.markdown('### Posts')
        st.write(df[['Date','Post']])
        
   
        
        
   
    

    
## Facebook Page

if rad == 'Facebook':
    with open('facebook.css') as f:
        set_png_as_page_bg('imgs/facebook_.gif')
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # loading the dataset
    #df1= pd.read_csv('C:/Users/Tiro/OneDrive/Documents/GitHub/internship-project-2207-14/Sentimental_Tweets.csv')
    dfacebook = pd.read_csv('C:/Users/Tiro/OneDrive/Documents/GitHub/internship-project-2207-14/ExploreAI social media data - FB (1).csv')
    
    #cleaning the Dataset with the imported NLP Funtion
    #Twitter_NLP.cleaning(df)
    
    #Filtering the cleaned dataset  with the imported Filtering Function
    #Filters.filter_dataframe(df1)
    df2 = Filters.filter_dataframe(dfacebook)
    
    # Row a
    likes= df2["Likes on posts"].sum()
    reach= df2["Post reach (page likers [fans] only)"].sum()
    impressions= df2["Post impressions (page likers [fans] only)"].sum()
    comments= df2["Comments on posts"].sum()
    shares= df2["Shares on posts"].sum()
    
    col1, col2, col3,col4,col5 = st.columns(5)
    col1.metric(label=" ### Reach", value=int(reach), delta="")
    col2.metric(label=" ### Engagements", value=int(impressions), delta="")
    col3.metric(label=" ### Likes", value=int(likes), delta="")
    col4.metric(label=" ### Comments", value=int(comments), delta="")
    col5.metric(label=" ### Shares", value=int(shares), delta="")
    
   
    
    #Row B
    c1, c2 = st.columns((7,3))    
    with c1:
        st.markdown('### Line graph')
        graph = df2[['Post creation date', 'Total post reactions', 'Likes on posts', 'Shares on posts', 'Comments on posts', 'Comments on shares', 'Post impressions (page likers [fans] only)','Post reach (page likers [fans] only)']].groupby(['Post creation date']).sum().reset_index()
        st.write(px.line(graph, x= 'Post creation date', y= 'Likes on posts',color_discrete_sequence=px.colors.qualitative.Antique))
     
    with c2:
        st.markdown('### Posts')
       
        st.write(df2[['Post creation date','Post message']])
        
    
    
    
##Instagram Page

if rad == 'Instagram':
    with open('insta.css') as f:
        set_png_as_page_bg('imgs/insta.JPG')
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    #Row A
    a1, a2, a3 = st.columns(3)
    a1.file_uploader('Upload File')
    a2.metric("Likes", "435", "")
    a3.metric("comments", "1230", "")
    
    # Row B
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("story likes", "4", "")
    b2.metric("Reach", "345", "")
    b3.metric("wordcloud", "30", "")
    b4.metric("hashTags", "6", "")
   
    
## Linkedin Page
   
if rad == 'LinkedIn':
   with open('facebook.css') as f:
       set_png_as_page_bg('imgs/linkedin.JPEG')
       st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
       
       # Row A
       a1, a2, a3 = st.columns(3)
       a1.file_uploader('Upload File')
       a2.metric("Likes", "435", "")
       a3.metric("comments", "1230", "")
       
       # Row B
       b1, b2, b3, b4 = st.columns(4)
       b1.metric("story likes", "4", "")
       b2.metric("Reach", "345", "")
       b3.metric("wordcloud", "30", "")
       b4.metric("hashTags", "6", "")
   
    