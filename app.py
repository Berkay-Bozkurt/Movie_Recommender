# import std libraries
import numpy as np
import pandas as pd
import requests
import os
from dotenv import load_dotenv

from IPython.display import HTML
import pickle
import json

import streamlit as st
from st_aggrid import AgGrid
from main import main
from data_transformation import df

def create_link(model, df):
        for i in range(len(model)):
            st.subheader(f"Movie {i+1}")
            load_dotenv()
            api_key = os.getenv('MOVIEDB_API_KEY')
            url=(f"https://api.themoviedb.org/3/movie/{df[df['title']==model[i]]['tmdbId'].unique()[0]}?api_key={api_key}&language=en-US")
            re=requests.get(url=url)
            re=re.json()
            col1,col2 = st.columns([1, 2])
            with col1:
                st.image(f"https://image.tmdb.org/t/p/w500/{re['poster_path']}")
            with col2:
                st.subheader(re["original_title"])
                st.caption(f"###### Genre: {re['genres'][0]['name']}, Year: {re['release_date'][:4]}, Language: {re['spoken_languages'][0]['english_name']}")
                st.write(re["overview"])
                st.text(f"Rating: {round(re['vote_average'],1)}")
                st.progress(float(re["vote_average"]/10))
                st.markdown('####')

BEST_MOVIES = pd.read_csv("best_movies.csv")
BEST_MOVIES.rename(
    index=lambda x: x+1,
    inplace=True
    )
TITLES = ["~~~"] + list(BEST_MOVIES['title'].sort_values()) 

# sidebar
with st.sidebar:
    # title
    st.title("It's movie time!")
    st.image('streamlit/movie_time.jpg')
    # blank space
    st.write("")
    page = st.selectbox(
        "what would you like?",
        [
            "welcome baby",
            "popular movies",
            "rate some movies",
            "recommended movies"
            ]
        ) 

if page == "welcome baby":
    # slogan
    st.write("""
    *Movies are like magic tricks (Jeff Bridges)*
    """)
    # blank space
    st.write("")
    # image
    st.image('streamlit/movie_pics.png')

##########################################################
# Popular Movies
##########################################################

elif page == "popular movies":
    # title
    st.title("Popular Movies")
    col1,col2,col3,col4 = st.columns([10,1,5,5])
    with col1:
        n = st.slider(
        label="how many movies?",
        min_value=1,
        max_value=10,
        value=5
        ) 
    with col3:
        st.markdown("####")
        genre = st.checkbox("include genres")
    with col4:
        st.markdown("###")
        show_button = st.button(label="show movies") 
    
    if genre:
        popular_movies = BEST_MOVIES[['movie_title','genres']]
    else:
        popular_movies = BEST_MOVIES[['movie_title']]

    st.markdown("###")
    if show_button:
        st.write(
            HTML(popular_movies.head(n).to_html(escape=False))
            )

##########################################################
# Rate Movies
##########################################################

elif page == "rate some movies":
    # title
    st.title("Rate Movies")
    #
    col1,col2,col3 = st.columns([10,1,5])
    with col1:
        m1 = st.selectbox("movie 1", TITLES)
        st.write("")
        m2 = st.selectbox("movie 2", TITLES)
        st.write("")
        m3 = st.selectbox("movie 3", TITLES)
        st.write("")
        m4 = st.selectbox("movie 4", TITLES)
        st.write("")
        m5 = st.selectbox("movie 5", TITLES) 
    
    with col3:
        r1 = st.slider(
            label="rating 1",
            min_value=1,
            max_value=5,
            value=3
            ) 
        r2 = st.slider(
            label="rating 2",
            min_value=1,
            max_value=5,
            value=3
            ) 
        r3 = st.slider(
            label="rating 3",
            min_value=1,
            max_value=5,
            value=3
            ) 
        r4 = st.slider(
            label="rating 4",
            min_value=1,
            max_value=5,
            value=3
            ) 
        r5 = st.slider(
            label="rating 5",
            min_value=1,
            max_value=5,
            value=3
            ) 

    query_movies = [m1,m2,m3,m4,m5]
    query_ratings = [r1,r2,r3,r4,r5]
    
    user_query = dict(zip(query_movies,query_ratings))

    # get user query
    st.markdown("###")
    user_query_button = st.button(label="save user query") 
    if user_query_button:
        json.dump(
            user_query,
            open("streamlit/user_query.json",'w')
            )
        st.write("")
        st.write("user query saved successfully")

##########################################################
# Movie Recommendations
##########################################################
else:
    # title
    st.title("Movie Recommendations")
    col1,col2,col3,col4,col5 = st.columns([1,5,1,5,1])
    with col2:
        recommender = st.radio(
            "recommender type",
            ["NMF Recommender","Neighborhood Recommender","Mix Recommender"]
            )
    with col4:
        st.write("###")
        recommend_button = st.button(label="recommed movies")

    #load user query
    if recommend_button:
        user_query = json.load(open("streamlit/user_query.json"))
        st.title("Recommendations For You")
        nmf, near, mix = main()

        if recommender == "NMF Recommender":
            create_link(nmf, df)
        elif recommender == "Neighborhood Recommender":
            create_link(near, df)
        elif recommender == "Mix Recommender":
            create_link(mix, df)