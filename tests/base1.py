import sys
sys.path.append("..")
import streamlit as st
# Page setting
st.set_page_config(layout="wide", initial_sidebar_state='expanded')
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import gspread_dataframe as gsdf
from src.data.make_dataset import get_base64_of_bin_file, set_png_as_page_bg
from Credentials import credentials
import Filters
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import Text_Cleaner
from collections import Counter
from datetime import datetime, date

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('stopwords')


    	    	    



#sidebar settings
st.sidebar.header('HMS `Dashboards`' )
    
## Home Page   
rad = st.sidebar.radio("Navigation", ['Home','Overall Dashboard', 'Twitter','Facebook', 'Instagram', 'LinkedIn'])

if rad == 'Home':   
   
   
   
   st.markdown("<h1 style='text-align: center; color: black;'>H.M.S</h1>", unsafe_allow_html=True)
   st.markdown("<h2 style='text-align: center; color: grey;'>Tracking And Analyzing Conversations And Trends Related To Your Brand</h2>", unsafe_allow_html=True)
   
   
   st.subheader('Instructions')                                                                                
   st.write('Towards the right hand side, you will find a navigation bar which contains all the social media dashboards this app has to offer.')
   st.write('On the navigation bar, you will find five dashboards avalable, `Summary`, `Twitter`, `Facebook`, `Instagram` and `LinkedIn`. Simply click on one and you will be directed to a dashboard.')
   st.write('On the dshboards, in the top left corner there is an `Add Filter` button. This button helps to filter out the whole dashbaord by which ever specific data you want. eg fliter by date or a specific post.')

   
   st.markdown('`Metrics Definition`')    
   plot_data = st.selectbox('`Select Metric`', ['Impressions', 'Reach', 'Engagements'])
   if plot_data == 'Impressions':
           st.write('`Impressions` is a metric that counts the number of times your content is displayed, regardless of whether it was interacted with or not. Impressions means that your content was delivered to your audiences` feed.')
           st.write('A person could have multiple impressions ')
           st.write('For example, a facebook post could show up on a persons news feed from the original publisher and appear again when a friend shares the publishers post. If a person saw both forms of activity on their feed, that counts as two impressions for the same post.')

   if plot_data == 'Reach':
           st.write('The `Reach` metric is the total number of unique people who see your content.This the metric that indicates to a business the number of people that about their Brand / Business.')
           st.write('On social media `reach` is the total number of unique users that see your conent on their feed.')   

   if plot_data == 'Engagements':
           st.write('`Engagement` metrics are metrics that indicate how users or site visitors, interact with your social media properties eg. Website, Social Media pages, Blogs so on and so forth. ')
           st.write('Some examples of engagement metrics are ~ Likes, comments, Post Shares, Page Views etc. The list can go on but these are just some basic engagement metrics.')
            




if rad == 'Overall Dashboard':

    
    # loading the Twitter dataset
    client = credentials.client
    sheet = client.open('ExploreAI social media data').worksheet('Twitter Sprout')
    dt1 = sheet.get_all_records()
    df_twitter = pd.DataFrame(dt1[1:], columns=dt1[0])
    
    
    # loading the Linkedin dataset
    
    client = credentials.client
    sheet = client.open('ExploreAI social media data').worksheet('LinkedIn')
    dt2 = sheet.get_all_records()
    df_linkedin = pd.DataFrame(dt2[1:], columns=dt2[0])
    
    

    # loading the instagram dataset
  
    client = credentials.client
    sheet = client.open('ExploreAI social media data').worksheet('Instagram')
    dt3 = sheet.get_all_records()
    df_instagram = pd.DataFrame(dt3[1:], columns=dt3[0])
    
    # loading the Facebook dataset
  
    client = credentials.client
    sheet = client.open('ExploreAI social media data').worksheet('FB')
    dt4 = sheet.get_all_records()
    df_facebook = pd.DataFrame(dt4[1:], columns=dt4[0])


   

    
    # Creating copies of dataframes
    twitter_df = df_twitter.copy()
    linkedin_df = df_linkedin.copy()
    instagram_df1 = df_instagram.copy()
    #facebook = df_facebook.copy()

    

    facebook1 = df_facebook.drop('Post comment text', axis=1)
    facebook_df2 = facebook1.drop_duplicates()

    # standardising names
    facebook_df2['Impressions']=facebook_df2['Post impressions (page likers [fans] only)']
    facebook_df2['Comments']=facebook_df2['Comments on posts']
    facebook_df2['Likes']=facebook_df2['Likes on posts']
    facebook_df2['Date']=facebook_df2['Post creation date']

    instagram_df1['Impressions']=instagram_df1['Media impressions']
    instagram_df1['Comments']=instagram_df1['Comments count']
    instagram_df1['Likes']=instagram_df1['Like count']
    instagram_df1['Date']=instagram_df1['Media created']

    # Change data to date type
    facebook_df2['Date'] = pd.to_datetime(facebook_df2['Date'])
    facebook_df2['Date'] = facebook_df2['Date'].dt.tz_localize(None)
    facebook_df = facebook_df2.sort_values(by='Date',ascending=False) #sorting facebook data by date column

    twitter_df['Date'] = pd.to_datetime(twitter_df['Date'])
    twitter_df['Date'] = twitter_df['Date'].dt.tz_localize(None)

    linkedin_df['Date'] = pd.to_datetime(linkedin_df['Date'])
    linkedin_df['Date'] = linkedin_df['Date'].dt.tz_localize(None)

    instagram_df1['Date'] = pd.to_datetime(instagram_df1['Date'])
    instagram_df1['Date'] = instagram_df1['Date'].dt.tz_localize(None)
    instagram_df = instagram_df1.sort_values(by='Date',ascending=False) #sorting instagram data by date column

    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # fixing columns
    cols_to_replace = ['Impressions','Likes', 'Comments']
    for col in cols_to_replace:
       
        facebook_df[col] = pd.to_numeric(facebook_df[col],errors='coerce')
        facebook_df[col] = facebook_df[col].replace('', 0) # replace empty strings with 0
        facebook_df[col] = facebook_df[col].astype(float) # convert column to float
        
       
        twitter_df[col] = pd.to_numeric(twitter_df[col],errors='coerce')
        twitter_df[col] = twitter_df[col].replace('', 0) # replace empty strings with 0
        twitter_df[col] = twitter_df[col].astype(float) # convert column to float
   
        
        linkedin_df[col] = pd.to_numeric(linkedin_df[col],errors='coerce')
        linkedin_df[col] = linkedin_df[col].replace('', 0) # replace empty strings with 0
        linkedin_df[col] = linkedin_df[col].astype(float) # convert column to float

       
        instagram_df[col] = pd.to_numeric(instagram_df[col],errors='coerce')
        instagram_df[col] = instagram_df[col].replace('', 0) # replace empty strings with 0
        instagram_df[col] = instagram_df[col].astype(float) # convert column to float
    

    st.sidebar.markdown("## Select metric below")
    select = st.sidebar.selectbox("Metric", ['Impressions','Likes','Comments'])

    # Date filter
    start_dt = st.sidebar.date_input('Start date', value=linkedin_df['Date'].min())
    end_dt = st.sidebar.date_input('End date', value=date.today())

    if start_dt <= end_dt:
        facebook_df = facebook_df[facebook_df['Date'] > datetime(start_dt.year, start_dt.month, start_dt.day)]
        facebook_df = facebook_df[facebook_df['Date'] < datetime(end_dt.year, end_dt.month, end_dt.day)]

        twitter_df = twitter_df[twitter_df['Date'] > datetime(start_dt.year, start_dt.month, start_dt.day)]
        twitter_df = twitter_df[twitter_df['Date'] < datetime(end_dt.year, end_dt.month, end_dt.day)]

        linkedin_df = linkedin_df[linkedin_df['Date'] > datetime(start_dt.year, start_dt.month, start_dt.day)]
        linkedin_df = linkedin_df[linkedin_df['Date'] < datetime(end_dt.year, end_dt.month, end_dt.day)]

        instagram_df = instagram_df[instagram_df['Date'] > datetime(start_dt.year, start_dt.month, start_dt.day)]
        instagram_df = instagram_df[instagram_df['Date'] < datetime(end_dt.year, end_dt.month, end_dt.day)]

        st.write()
    else:
        st.error('Start date must be > End date')

    st.title('Summary Overview of '+' '+ select)

    # Metrics
    Total_metric =facebook_df[select].sum() + twitter_df[select].sum() + linkedin_df[select].sum() + instagram_df[select].sum()
    facebook_metric = (facebook_df[select].sum()/Total_metric)*100
    twitter_metric = (twitter_df[select].sum()/Total_metric)*100
    linkedin_metric = (linkedin_df[select].sum()/Total_metric)*100
    instagram_metric = (instagram_df[select].sum()/Total_metric)*100
    
      
    # Creating columns of a dataframe
    Social_Media = ['Facebook', 'Twitter', 'LinkedIn', 'Instagram']
    Selected = [facebook_metric,twitter_metric,linkedin_metric, instagram_metric]
    
    # Values for cards    
    metric_fb = facebook_df[select].sum()
    metric_twt = twitter_df[select].sum()
    metric_ln = linkedin_df[select].sum()
    metric_in = instagram_df[select].sum()
    Selected = [metric_fb, metric_twt, metric_ln, metric_in]
    data = pd.DataFrame(dict (Social_Media = Social_Media, Selected = Selected))

    ##################### Layout Application ##################
    # First row
    col1, col2, col3, col4 = st.columns([2,2,2,2])
    col1.metric(label="Facebook "+ " "+ select, value=int(metric_fb), delta="")
    col2.metric(label="Twitter "+ " "+ select, value=int(metric_twt), delta="")
    col3.metric(label="LinkedIn "+ " "+ select, value=int(metric_ln), delta="")
    col4.metric(label="Instagram "+ " "+ select, value=int(metric_in), delta="")

    # Second Row
    col1, col2 = st.columns([1,1])
    with col1:
        fig1 = px.pie(data, values='Selected', names='Social_Media', hole=0.4) 
        fig1.update_layout(title={'text': 'Distribution of'+' '+ select + ' ' +'in social media platforms','x':0.5,'xanchor': 'center', 'yanchor': 'top'})
        st.plotly_chart(fig1, use_container_width=True)

    with col2:        
        socio_media = ['Facebook','Twitter','LinkedIn','Instagram']
        options = st.sidebar.multiselect('Compare your social media platform', options = socio_media, default=socio_media)
        
        
        fig = go.Figure()

        if 'Facebook' in options:
            fig.add_trace(go.Scatter(
                x=facebook_df['Date'],
                y=facebook_df[select],
                name='Facebook',
                mode='lines'
            ))

        if 'Twitter' in options:
            fig.add_trace(go.Scatter(
                x=twitter_df['Date'],
                y=twitter_df[select],
                name='Twitter',
                mode='lines'
            ))

        if 'LinkedIn' in options:
            fig.add_trace(go.Scatter(
                x=linkedin_df['Date'],
                y=linkedin_df[select],
                name='LinkedIn',
                mode='lines'
            ))
     

        if 'Instagram' in options:
            fig.add_trace(go.Scatter(
                x=instagram_df['Date'],
                y=instagram_df[select],
                name='Instagram',
                mode='lines'
            ))

        fig.update_layout(
            title={'text': select +' '+'Over Time','y':0.9,'x':0.5,'xanchor': 'center', 'yanchor': 'top'},
            xaxis=dict(title='Date'),
            yaxis=dict(title= 'Impressions')
        )
        st.plotly_chart(fig, use_container_width=True)

######################################### End of Summary Page #################################################    	    	    

   
    
## Twitter Page
   
if rad == 'Twitter':
    st.title('Twitter')
    #with open('style.css') as f:
        #st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
   

    #set_png_as_page_bg('imgs/twit.webp') 
    
    # loading the dataset
    client = credentials.client
    sheet = client.open('ExploreAI social media data').worksheet('Twitter Sprout')
    dt = sheet.get_all_records()
    df1 = pd.DataFrame(dt[1:], columns=dt[0])

    
    # Convert the Date column to a datetime data type
    df1['Date'] = pd.to_datetime(df1['Date'])
    
    # remove empty cells in video views
    df1['Video Views'] = df1['Video Views'].replace('', 0)
    
    # convert columns to float
    cols_to_replace = ['Engagements', 'Impressions','Engagement Rate (per Impression)', 'Organic Impressions', 'Potential Reach', 'Reactions', 'Likes', 'Comments', 'Shares', 'Post Link Clicks', 'Other Post Clicks', 'Post Clicks (All)', 'Other Engagements']

    for col in cols_to_replace:
        df1[col] = df1[col].replace('%', '', regex=True) # remove percentage sign
        df1[col] = df1[col].replace('', 0) # replace empty strings with 0
        df1[col] = df1[col].astype(float) # convert column to float

    # convert columns to string
    df1 = df1.astype({"Post ID": float, "Network": str, "Post Type": str, "Content Type": str, "Profile": str,
                      "Link":str, "Post": str, "Linked Content": str})
    

   
    df = Filters.filter_dataframe(df1)
    
    groupedby_postid = df[['Post ID', 'Likes', 'Comments', 'Engagements',
                           'Impressions', 'Shares', 'Potential Reach']].groupby(['Post ID']).sum(numeric_only=True).reset_index()
    
    groupedby_date= df[['Date','Impressions', 'Likes', 'Engagements', 'Comments','Shares', 'Potential Reach']].groupby(['Date']).sum(numeric_only=True).reset_index().sort_values(['Date'], axis=0, ascending=True)
    
    likes= groupedby_postid["Likes"].sum()
    comments= groupedby_postid["Comments"].sum()
    engagements= groupedby_postid["Engagements"].sum()
    impressions = groupedby_postid['Impressions'].sum()
    shares = groupedby_postid['Shares'].sum()
    reach = groupedby_postid['Potential Reach'].sum()
    
    
    # Row A
    col1, col2, col3, col4, col5, col6  = st.columns(6)
    
    col1.metric("Likes", np.array(likes).astype(int), '{}%'.format(round((np.array(groupedby_date['Likes'])[-1] - groupedby_date['Likes'].max()) / groupedby_date['Likes'].max() * 100, 2)))
    col2.metric("Comments", np.array(comments).astype(int), '{}%'.format(round((np.array(groupedby_date['Comments'])[-1] - groupedby_date['Comments'].max()) / groupedby_date['Comments'].max() * 100, 2)))
    col3.metric("Shares", np.array(shares).astype(int), '{}%'.format(round((np.array(groupedby_date['Shares'])[-1] - groupedby_date['Shares'].max()) / groupedby_date['Shares'].max() * 100, 2)))   
    col4.metric("Engagements", np.array(engagements).astype(int), '{}%'.format(round((np.array(groupedby_date['Engagements'])[-1] - groupedby_date['Engagements'].max()) / groupedby_date['Engagements'].max() * 100, 2)))
    col5.metric("Impressions", np.array(impressions).astype(int), '{}%'.format(round((np.array(groupedby_date['Impressions'])[-1] - groupedby_date['Impressions'].max()) / groupedby_date['Impressions'].max() * 100, 2)))
    col6.metric("Potential Reach", np.array(reach).astype(int), '{}%'.format(round((np.array(groupedby_date['Potential Reach'])[-1] - groupedby_date['Potential Reach'].max()) / groupedby_date['Potential Reach'].max() * 100, 2)))

    
    
   
    #Row B
    
    df_num = df[['Likes', 'Comments', 'Engagements',
                           'Impressions', 'Shares', 'Potential Reach']]
    
    df_corr = df_num.corr()
    fig1 = px.imshow(df_corr, text_auto=True, color_continuous_scale=px.colors.sequential.thermal, zmin= -1, zmax= 1)
    fig1.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig1.update_layout(title_text='Correlation Table')
    fig2 = px.scatter(df_num, x='Impressions', y='Engagements', color='Likes', color_continuous_scale=px.colors.sequential.thermal)
    fig2.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig2.update_layout(title_text='Impressions Influence on Engagements')
    
    a1, a2 = st.columns(2)
    a1.plotly_chart(fig1, use_container_width=True)
    a2.plotly_chart(fig2, use_container_width=True)
    
    
    #Row C
    
    b1, b2 = st.columns((7,3))
    with b1:
        plot_data = st.sidebar.selectbox('Select data', ['Likes', 'Comments','Shares','Engagements','Impressions','Potential Reach'])
        
        grp= df[['Date','Impressions', 'Likes', 'Engagements', 'Comments','Potential Reach', 'Shares']].groupby(['Date']).sum(numeric_only=True).reset_index()
        
        fig3 = px.line(grp, x= 'Date', y= plot_data, markers=True)
        fig3.update_layout(title_text='Time Series')
        if plot_data == 'Likes':
            fig3.update_traces(line_color='#47A992')
        elif plot_data == 'Shares':
            fig3.update_traces(line_color='#87CBB9')
        elif plot_data == 'Comments':
            fig3.update_traces(line_color='#227C70')
        elif plot_data == 'Engagements':
            fig3.update_traces(line_color='#439A97')
        elif plot_data == 'Impressions':
            fig3.update_traces(line_color='#FFB4B4')
        else:
            fig3.update_traces(line_color='#C37B89')
        st.plotly_chart(fig3, use_container_width=True)
        
     #color_discrete_sequence=px.colors.qualitative.Plotly_r
        
    with b2:
        # PIe chart for Twitter
        
        labels = ['Likes','Shares','Comments']
        values = [df["Likes"].sum(), df["Shares"].sum(), df["Comments"].sum()]
        fig4 = px.pie(df, values=values, names=labels,color=labels, color_discrete_map={'Likes': '#47A992',
                                                                           'Shares': '#87CBB9',
                                                                           'Comments': '#227C70'}, hole=0.4)
        fig4.update_layout(title_text='Distribution of Engagements')
        b2.plotly_chart(fig4, use_container_width=True)

        
      
    # Row D
    
    c1, c2 = st.columns((2,1))
    df['clean_text'] = Text_Cleaner.clean_text(df['Post'])
    clean_text_twitter = ' '.join(df['clean_text'])
    wordcloud_twitter = WordCloud(width=800, 
                           height=500, 
                           random_state=110, 
                           max_font_size=110, 
                           background_color='#F9F9F9',
                           colormap="Greens").generate(clean_text_twitter)
    
    fig5 = plt.figure()
    ax1 = fig5.add_subplot(211)
    ax1.imshow(wordcloud_twitter,interpolation='bilinear')
    ax1.axis("off")
    ax1.set_title('Tweets\' Topic Cloud', fontsize=8)
    
    c1.pyplot(fig5)
    
    
    counts_tweet = Counter(' '.join(df['clean_text']).split()).most_common(25)
    counts_tweet = pd.DataFrame(counts_tweet, columns=['word', 'frequency']).head(5)
    
    fig6 = px.bar(counts_tweet, x='word', y='frequency', color='word', color_discrete_map={'data': '#609966',
                                                                                           'science': '#9DC08B',
                                                                                           'learning': '#98D8AA',
                                                                                           'students': '#BEF0CB',
                                                                                           'know': '#C7E8CA'})
    fig6.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig6.update_layout(title_text='Popular Topics from Tweets')
    c2.plotly_chart(fig6, use_container_width=True)
    
    
    
    
    # Row E
    
    tweet_tags = Text_Cleaner.hashtag_extract(df['Post'].str.lower(), 10)
    fig7 = px.funnel(tweet_tags, y='hashtag', x='count', color='hashtag')
    fig7.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig7.update_layout(title_text='Popular Hashtags From Tweets')
    fig7.update_layout(legend_traceorder="reversed")
    st.plotly_chart(fig7, use_container_width=True)
    
    
    
    
    
        
      
        
      



## Facebook Page

if rad == 'Facebook':
    st.title('Facebook')
    #with open('facebook.css') as f:
        
        #set_png_as_page_bg('imgs/_SocialParty14_small.webp')
        
        #st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # loading the dataset
    client = credentials.client
    sheet = client.open('ExploreAI social media data').worksheet('FB')
    values = sheet.get_all_records()
    dfacebook = pd.DataFrame(values[1:], columns=values[0])
    dfacebook['Shares on posts'].loc[dfacebook['Shares on posts'] == ''] = 0
    dfacebook['Post link clicks'].loc[dfacebook['Post link clicks'] == ''] = 0
    dfacebook['Total post reactions'].loc[dfacebook['Total post reactions'] == ''] = 0
    dfacebook['Post comment text'].loc[dfacebook['Post comment text'] == ''] = ' '
    #dfacebook = pd.read_csv('../ExploreAI social media data - FB (1).csv')
    #st.dataframe(dfacebook)
    
    dfacebook[['Likes on posts', 'Post reach (page likers [fans] only)', 
                           'Post impressions (page likers [fans] only)', 'Comments on posts', 'Shares on posts', 'Post link clicks', 'Total post reactions']] = dfacebook[['Likes on posts', 'Post reach (page likers [fans] only)', 
                           'Post impressions (page likers [fans] only)', 'Comments on posts', 'Shares on posts', 'Post link clicks', 'Total post reactions']].astype(int)
    
    
    dfacebook['Post comment text'] = dfacebook['Post comment text'].astype(str)
    df2 = Filters.filter_dataframe(dfacebook)
    
    
    grouped_by_post = df2[['Post ID','Likes on posts', 'Post reach (page likers [fans] only)', 
                           'Post impressions (page likers [fans] only)', 'Comments on posts', 'Shares on posts', 'Post link clicks', 'Total post reactions']].groupby(['Post ID']).sum().reset_index().sort_values(['Post ID'], axis=0, ascending=False)
    
    # Row A
    likes= grouped_by_post["Likes on posts"].sum()
    reach= grouped_by_post["Post reach (page likers [fans] only)"].sum()
    impressions= grouped_by_post["Post impressions (page likers [fans] only)"].sum()
    comments= grouped_by_post["Comments on posts"].sum()
    shares= grouped_by_post["Shares on posts"].sum()
    clicks= grouped_by_post['Post link clicks'].sum()
    reactions= grouped_by_post['Total post reactions'].sum()
    
    col1, col2, col3,col4,col5,col6,col7 = st.columns(7)
    col1.metric("Reach", np.array(reach), '{}%'.format(round((np.array(grouped_by_post['Post reach (page likers [fans] only)'])[-1] - grouped_by_post['Post reach (page likers [fans] only)'].max()) / grouped_by_post['Post reach (page likers [fans] only)'].max() * 100, 2)))
    col2.metric("Engagements", np.array(impressions), '{}%'.format(round((np.array(grouped_by_post['Post impressions (page likers [fans] only)'])[-1] - grouped_by_post['Post impressions (page likers [fans] only)'].max()) / grouped_by_post['Post impressions (page likers [fans] only)'].max() * 100, 2)))
    col3.metric("Likes", np.array(likes), '{}%'.format(round((np.array(grouped_by_post['Likes on posts'])[-1] - grouped_by_post['Likes on posts'].max()) / grouped_by_post['Likes on posts'].max() * 100, 2)))
    col4.metric("Comments", np.array(comments), '{}%'.format(round((np.array(grouped_by_post['Comments on posts'])[-1] - grouped_by_post['Comments on posts'].max()) / grouped_by_post['Comments on posts'].max() * 100, 2)))
    col5.metric("Shares", np.array(shares), '{}%'.format(round((np.array(grouped_by_post['Shares on posts'])[-1] - grouped_by_post['Shares on posts'].max()) / grouped_by_post['Shares on posts'].max() * 100, 2)))
    col6.metric('Clicks', np.array(clicks), '{}%'.format(round((np.array(grouped_by_post['Post link clicks'])[-1] - grouped_by_post['Post link clicks'].max()) / grouped_by_post['Post link clicks'].max() * 100, 2)))
    col7.metric('Reactions', np.array(reactions), '{}%'.format(round((np.array(grouped_by_post['Total post reactions'])[-1] - grouped_by_post['Total post reactions'].max()) / grouped_by_post['Total post reactions'].max() * 100, 2)))
    
    
    # Row B
    df_num = df2[['Likes on posts', 'Post reach (page likers [fans] only)', 'Post impressions (page likers [fans] only)', 
                  'Comments on posts', 'Shares on posts', 'Post link clicks', 'Total post reactions']]
    df_num.rename(columns={'Likes on posts': 'Likes',
                           'Post reach (page likers [fans] only)': 'Reach', 
                           'Post impressions (page likers [fans] only)': 'Impressions',
                           'Comments on posts': 'Comments', 
                           'Shares on posts': 'Shares',
                           'Post link clicks': 'Clicks', 
                           'Total post reactions': 'Reactions'}, inplace=True)
    df_corr = df_num.corr()
    fig1 = px.imshow(df_corr, text_auto=True, color_continuous_scale=px.colors.sequential.thermal, zmin= -1, zmax= 1)
    fig1.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig1.update_layout(title_text='Correlation Table')
    fig2 = px.scatter(df_num, x='Impressions', y='Reach', color='Clicks', color_continuous_scale=px.colors.sequential.thermal)
    fig2.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig2.update_layout(title_text='Impressions Influence on Reach')
    
    a1, a2 = st.columns(2)
    a1.plotly_chart(fig1, use_container_width=True)
    a2.plotly_chart(fig2, use_container_width=True)
    

  

        
       
    
    
    #Row C
    b1, b2 = st.columns((7,3))  
    
    with b2:
   
       
          
        labels = ['Likes','Shares','Comments']
        values = [df2["Likes on posts"].sum(), df2["Shares on posts"].sum(), df2["Comments on posts"].sum()]
        
        fig4 = px.pie(df2, values=values, names=labels, color=labels, color_discrete_map={'Likes': '#80558C',
                                                                                          'Shares': '#CD5888',
                                                                                          'Comments': '#C689C6'},
                      hole=0.4)
        fig4.update_layout(title_text='Distribution of Engagements')
        fig4.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
        b2.plotly_chart(fig4, use_container_width=True)
        
    with b1:
        #st.markdown('### Time series')
        st.sidebar.write( "Select graph data")
        plo_data = st.sidebar.selectbox('Select data', ['Likes on posts', 'Comments on posts',"Post reach (page likers [fans] only)","Post impressions (page likers [fans] only)","Shares on posts"])
        
        graph = df2[['Post creation date', 'Total post reactions', 'Likes on posts', 'Shares on posts', 'Comments on posts', 'Comments on shares', 'Post impressions (page likers [fans] only)','Post reach (page likers [fans] only)']].groupby(['Post creation date']).sum().reset_index()
        fig3 = px.line(graph, x= 'Post creation date', y= plo_data, markers=True)
        fig3.update_layout(title_text='Time Series')
        fig3.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
        if plo_data == 'Likes on posts':
            fig3.update_traces(line_color='#80558C')
        elif plo_data == 'Shares on posts':
            fig3.update_traces(line_color='#CD5888')
        elif plo_data == 'Comments on posts':
            fig3.update_traces(line_color='#C689C6')
        elif plo_data == 'Post reach (page likers [fans] only)':
            fig3.update_traces(line_color='#852999')
        else:
            fig3.update_traces(line_color='#80489C')
        st.plotly_chart(fig3, use_container_width=True)
    
    
    
    
    # Row D
    
    c1, c2 = st.columns((2,1))
    df2['clean_text'] = Text_Cleaner.clean_text(df2['Post comment text'])
    clean_text_fb = ' '.join(df2['clean_text'])
    wordcloud_fb = WordCloud(max_words=100,
                             width=800, 
                           height=500, 
                           random_state=110, 
                           max_font_size=110, 
                           background_color='#F9F9F9',
                           colormap="Purples").generate(clean_text_fb)
    
    fig4 = plt.figure()
    ax1 = fig4.add_subplot(211)
    ax1.imshow(wordcloud_fb,interpolation='bilinear')
    ax1.axis("off")
    ax1.set_title('Facebook Comments\' Topic Cloud', fontsize=8)
    
    c1.pyplot(fig4)
    




    counts_fb = Counter(' '.join(df2['clean_text']).split()).most_common(25)
    counts_fb = pd.DataFrame(counts_fb, columns=['word', 'frequency']).head(5)
    
    fig5 = px.bar(counts_fb, x='word', y='frequency', color='word', color_discrete_map= {'data': '#AD7BE9',
                                                                                        'science': '#D9ACF5',
                                                                                        'explore': '#C3ACD0',
                                                                                        'course': '#FFCEFE',
                                                                                        'like': '#FDEBED'})
    fig5.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig5.update_layout(title_text='Popular Topics from Comments')
    c2.plotly_chart(fig5, use_container_width=True)






    # Row E
    
    d1, d2 = st.columns((2,1))
    df2['clean_text1'] = Text_Cleaner.clean_text(df2['Post message'])
    clean_text_fb1 = ' '.join(df2['clean_text1'])
    wordcloud_fb1 = WordCloud(max_words=100,
                              width=800, 
                           height=500, 
                           random_state=110, 
                           max_font_size=110, 
                           background_color='#F9F9F9',
                           colormap="Purples").generate(clean_text_fb1)
    
    fig6 = plt.figure()
    ax1 = fig6.add_subplot(211)
    ax1.imshow(wordcloud_fb1,interpolation='bilinear')
    ax1.axis("off")
    ax1.set_title('Facebook Posts\' Topic Cloud', fontsize=8)
    
    d1.pyplot(fig6)
    




    counts_fb1 = Counter(' '.join(df2['clean_text1']).split()).most_common(25)
    counts_fb1 = pd.DataFrame(counts_fb1, columns=['word', 'frequency']).head(5)
    
    fig7 = px.bar(counts_fb1, x='word', y='frequency', color='word', color_discrete_map={'data': '#AD7BE9',
                                                                                         'science': '#D9ACF5',
                                                                                         'engineering': '#C3ACD0',
                                                                                         'link': '#FFCEFE',
                                                                                         'apply': '#FDEBED'})
    fig7.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig7.update_layout(title_text='Popular Topics from Posts')
    d2.plotly_chart(fig7, use_container_width=True)
    
    
    
    
    
    
    
    
    
    
    
    




    
##Instagram Page

if rad == 'Instagram':
    st.title('Instagram')
    #with open('insta.css') as f:
        #set_png_as_page_bg('imgs/_SocialParty16_small.webp')
        #st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    client = credentials.client
    sheet = client.open('ExploreAI social media data').worksheet('Instagram')   # Opening the specified worksheet from the Google Sheet
    
    values = sheet.get_all_records()

    # Convert the values to a DataFrame
    dfinsta = pd.DataFrame(values[1:], columns=values[0])
    # Drop any rows with all NaN values
    #dfinsta.dropna(how='all', inplace=True)

    dfinsta['Media impressions'].loc[dfinsta['Media impressions'] == ''] = 0
    
    #dfinsta = np.array(dfinsta)[dfinsta.isna()] = 0.0
    
    
    #st.dataframe(dfinsta)
    
    dfinsta[['Like count', 'Comments count', 
             'Media impressions', 'Media reach']] = dfinsta[['Like count', 'Comments count', 
                                                             'Media impressions', 'Media reach']].astype(int)
    
    df1 = Filters.filter_dataframe(dfinsta)
    
    grouped_by_post = df1[['Media created','Like count', 'Comments count', 
                           'Media impressions', 'Media reach']].groupby(['Media created']).sum().reset_index().sort_values(['Media created'], axis=0, ascending=False)

    df_num = df1[['Like count', 'Comments count', 
                           'Media impressions', 'Media reach']]
    df_corr = df_num.corr()
    fig1 = px.imshow(df_corr, text_auto=True, color_continuous_scale=px.colors.sequential.thermal, zmin= -1, zmax= 1)
    fig1.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig1.update_layout(title_text='Correlation Table')
    fig2 = px.scatter(df_num, x='Media impressions', y='Media reach', color='Like count', color_continuous_scale=px.colors.sequential.thermal)
    fig2.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig2.update_layout(title_text='Impressions Influence on Reach')
    
    #Row A
    a1, a2, a3, a4 = st.columns(4)
    likes1 = grouped_by_post['Like count'].sum()
    comments1 = df1['Comments count'].sum()
    impressions1 = df1['Media impressions'].sum()
    reach1 = df1['Media reach'].sum()
    
    a1.metric('Likes', np.array(likes1), '{}%'.format(round((np.array(grouped_by_post['Like count'])[-1] - grouped_by_post['Like count'].max()) / grouped_by_post['Like count'].max() * 100, 2)))
    a2.metric("Comments", np.array(comments1), '{}%'.format(round((np.array(grouped_by_post['Comments count'])[-1] - grouped_by_post['Comments count'].max()) / grouped_by_post['Comments count'].max() * 100, 2)))
    a3.metric("Impressions", np.array(impressions1), '{}%'.format(round((np.array(grouped_by_post['Media impressions'])[-1] - grouped_by_post['Media impressions'].max()) / grouped_by_post['Media impressions'].max() * 100, 2)))
    a4.metric('Reach', np.array(reach1), '{}%'.format(round((np.array(grouped_by_post['Media reach'])[-1] - grouped_by_post['Media reach'].max()) / grouped_by_post['Media reach'].max() * 100, 2)))
    
    
    # Row B
    
    b1, b2 = st.columns(2)
    
    b1.plotly_chart(fig1, use_container_width=True)
    b2.plotly_chart(fig2, use_container_width=True)
    
    # Row C
    insta_plot = st.sidebar.selectbox('Select data', ['Like count', 'Comments count', 'Media impressions', 'Media reach'])
    
    fig3 = px.line(grouped_by_post, x='Media created', y=insta_plot, markers=True)
    fig3.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig3.update_layout(title_text='Time Series')
    if insta_plot == 'Like count':
        fig3.update_traces(line_color='#FF6D60')
    elif insta_plot == 'Comments count':
        fig3.update_traces(line_color='#F7D060')
    elif insta_plot == 'Media impressions':
        fig3.update_traces(line_color='#F3E99F')
    else:
        fig3.update_traces(line_color='#FD8A8A')
    st.plotly_chart(fig3, use_container_width=True)
    
    
    # Row D
    c1, c2 = st.columns((2,1))
    df1['clean_text'] = Text_Cleaner.clean_text(df1['Media caption'])
    clean_text_insta = ' '.join(df1['clean_text'])
    wordcloud_insta = WordCloud(max_words=100,
                                width=800, 
                           height=500, 
                           random_state=110, 
                           max_font_size=110, 
                           background_color='#F9F9F9',
                           colormap="Reds").generate(clean_text_insta)
    
    fig3 = plt.figure()
    ax1 = fig3.add_subplot(211)
    ax1.imshow(wordcloud_insta,interpolation='bilinear')
    ax1.axis("off")
    ax1.set_title('Instagram Posts\' Topic Cloud', fontsize=8)
    
    c1.pyplot(fig3)
    
    
    
    
    counts_insta = Counter(' '.join(df1['clean_text']).split()).most_common(25)
    counts_insta = pd.DataFrame(counts_insta, columns=['word', 'frequency']).head(5)
    
    fig4 = px.bar(counts_insta, x='word', y='frequency', color='word', color_discrete_map={'data': '#DC3535',
                                                                                           'science': '#FF5858',
                                                                                           'link': '#E97777',
                                                                                           'bio': '#FF9F9F',
                                                                                           'explorers': '#FCDDB0'})
    fig4.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig4.update_layout(title_text='Popular Topics from Captions')
    c2.plotly_chart(fig4, use_container_width=True)


    # Row E

    tags = Text_Cleaner.hashtag_extract(df1['Media caption'].str.lower(), 10)
    fig5 = px.funnel(tags, y='hashtag', x='count', color='hashtag')
    fig5.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig5.update_layout(title_text='Popular Hashtags on Instagram Captions')
    fig5.update_layout(legend_traceorder="reversed")
    st.plotly_chart(fig5, use_container_width=True)
    








    
## Linkedin Page
   
if rad == 'LinkedIn':
    st.title('LinkedIn')
    #with open('linkedin.css') as f:
        #set_png_as_page_bg('imgs/_SocialParty17_small.webp')
        #st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    client = credentials.client
    sheet = client.open('ExploreAI social media data').worksheet('LinkedIn')
    dt = sheet.get_all_records()
    df_linkedin = pd.DataFrame(dt[1:], columns=dt[0])
    
    df_linkedin = Filters.filter_dataframe(df_linkedin)
    
    line_plot = df_linkedin[['Date', 'Impressions', 'Clicks', 'Shares', 'Comments', 'Engagements', 'Likes']].groupby(['Date']).sum().reset_index().sort_values(['Date'], axis=0, ascending=True)
    line_plot['Click-through_rate'] = round(line_plot['Clicks'] / line_plot['Impressions'] *100 , 2)  
    
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    a1, a2 = st.columns(2)

    
    
    # Row A
    b4.metric("Likes", df_linkedin['Likes'].sum(), '{}%'.format(round((np.array(line_plot['Likes'])[-1] - line_plot['Likes'].max()) / line_plot['Likes'].max() * 100, 2)))
    b6.metric("Comments", df_linkedin['Comments'].sum(), '{}%'.format(round((np.array(line_plot['Comments'])[-1] - line_plot['Comments'].max()) / line_plot['Comments'].max() * 100, 2)))
    b2.metric("Engagements", df_linkedin['Engagements'].sum(), '{}%'.format(round((np.array(line_plot['Engagements'])[-1] - line_plot['Engagements'].max()) / line_plot['Engagements'].max() * 100, 2)))
    b3.metric("Clicks", df_linkedin['Clicks'].sum(), '{}%'.format(round((np.array(line_plot['Clicks'])[-1] - line_plot['Clicks'].max()) / line_plot['Clicks'].max() * 100, 2)))
    b1.metric('Impressions', df_linkedin['Impressions'].sum(), '{}%'.format(round((np.array(line_plot['Impressions'])[-1] - line_plot['Impressions'].max()) / line_plot['Impressions'].max() * 100, 2)))
    b5.metric('Shares', df_linkedin['Shares'].sum(), '{}%'.format(round((np.array(line_plot['Shares'])[-1] - line_plot['Shares'].max()) / line_plot['Shares'].max() * 100, 2)))
    
    
    
    # Row B
    df_num = df_linkedin[['Impressions', 'Clicks', 'Likes', 'Comments', 'Shares', 'Engagements', 'Engagement rate (%)']]
    df_corr = df_num.corr()
    fig1 = px.imshow(df_corr, text_auto=True, color_continuous_scale=px.colors.sequential.thermal, zmin= -1, zmax= 1)
    fig1.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig1.update_layout(title_text='Correlation Table')
    fig2 = px.scatter(df_num, x='Impressions', y='Engagements', color='Clicks', color_continuous_scale=px.colors.sequential.thermal)
    fig2.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig2.update_layout(title_text='Impressions Influence on Engagements')
    a1.plotly_chart(fig1, use_container_width=True)
    a2.plotly_chart(fig2, use_container_width=True)
    
    
    
    # Row C
    c1, c2 = st.columns((7,3)) 
    opts = st.sidebar.selectbox('Select data by Date', ('Impressions', 'Engagements','Clicks', 'Likes', 'Shares', 'Comments', 'Click-through_rate'))
    fig3 = px.line(line_plot, x='Date', y=opts, markers=True, color_discrete_sequence=px.colors.qualitative.Dark24_r)
    fig3.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig3.update_layout(title_text='Time Series')
    if opts == 'Clicks':
        fig3.update_traces(line_color='#19376D')
    elif opts == 'Likes':
        fig3.update_traces(line_color='#576CBC')
    elif opts == 'Shares':
        fig3.update_traces(line_color='#4E31AA')
    elif opts == 'Comments':
        fig3.update_traces(line_color='#62CDFF')
    elif opts == 'Impressions':
        fig3.update_traces(line_color='#6886C5')
    elif opts == 'Engagements':
        fig3.update_traces(line_color='#4F98CA')
    else:
        fig3.update_traces(line_color='#226597')
    c1.plotly_chart(fig3, use_container_width=True)
    
    
    names = ['Clicks', 'Likes', 'Shares', 'Comments']
    values = [line_plot['Clicks'].sum(), line_plot['Likes'].sum(), 
              line_plot['Shares'].sum(), line_plot['Comments'].sum()]
    
    fig5 = px.pie(line_plot, names=names, values=values, color=names, color_discrete_map={'Clicks': '#19376D',
                                                                                          'Likes': '#576CBC',
                                                                                          'Shares': '#4E31AA',
                                                                                          'Comments': '#62CDFF'},
                  hole=0.4)
    fig5.update_layout(title_text='Distribution of Engagements')
    fig5.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    c2.plotly_chart(fig5, use_container_width=True)
    
    
    # Row D
    d1, d2 = st.columns((2,1))
    df_linkedin['clean_text'] = Text_Cleaner.clean_text(df_linkedin['Update content text'])
    clean_text = ' '.join(df_linkedin['clean_text'])
    wordcloud_linkedin = WordCloud(max_words=100,
                                   width=800, 
                           height=500, 
                           random_state=110, 
                           max_font_size=110, 
                           background_color='#F9F9F9',
                           colormap="Blues").generate(clean_text)
    
    fig6 = plt.figure()
    ax1 = fig6.add_subplot(211)
    ax1.imshow(wordcloud_linkedin,interpolation='bilinear')
    ax1.axis("off")
    ax1.set_title('LinkedIn Posts\' Topic Cloud', fontsize=8)
    
    d1.pyplot(fig6)
    
    
    
    counts = Counter(' '.join(df_linkedin['clean_text']).split()).most_common(25)
    counts = pd.DataFrame(counts, columns=['word', 'frequency']).head(5)
    
    
    
    
    fig7 = px.bar(counts, x='word', y='frequency', color='word', color_discrete_map={'data': '#161D6F',
                                                                                     'science': '#005A8D',
                                                                                     'learning': '#1687A7',
                                                                                     'explore': '#98DED9',
                                                                                     'skills': '#CCF2F4'})
    fig7.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig7.update_layout(title_text='Popular Topics from Posts')
    d2.plotly_chart(fig7, use_container_width=True)
    
    
    
    # Row E
    
    
    
    tags = Text_Cleaner.hashtag_extract(df_linkedin['Update content text'].str.lower(), 10)
    fig4 = px.funnel(tags, x='count', y='hashtag' ,color='hashtag')
    fig4.update_layout({'plot_bgcolor': '#F9F9F9',
                        'paper_bgcolor': '#F9F9F9'})
    fig4.update_layout(title_text='Popular Hashtags on LinkedIn Posts')
    fig4.update_layout(legend_traceorder="reversed")
    st.plotly_chart(fig4, use_container_width=True)
   
