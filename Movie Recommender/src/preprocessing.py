# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 20:13:15 2022

@author: ASUS
"""


import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer


movies=pd.read_csv("ml-1m/movies.dat",sep='::',header=None, names=['movie_id', 'title', 'genre'])
movies['genre']=movies['genre'].apply(lambda x: x.replace('|',' '))
movies.head()

ratings=pd.read_csv("ml-1m/ratings.dat",sep='::',header=None)
ratings.columns =['user_id', 'movie_id', 'rating', 'timestamp'] 
ratings.head()

userdata=pd.read_csv("ml-1m/users.dat",sep='::',header=None)
userdata.head()
userdata.columns=["user_id","gender","age_group","occupation","zipcode"]
userdata.to_csv('users.csv',index=False)


from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin 

def first_search_result(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    first_result = soup.find('td', 'result_text').find('a').get('href')

    return first_result
def movie_description(next_page_url2):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(next_page_url2, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    #return soup.find("div", class_="summary_text").contents[0].strip()  # old version
    return soup.find("span", class_="sc-16ede01-1 kgphFu").get_text().strip()
def scrapeDescription(movieName):

    movieName = movieName.replace(' ','+')
    url = "http://www.imdb.com/find?ref_=nv_sr_fn&q=" + movieName + '&s=all'
    try:
        next_page = first_search_result(url)
    except:
        print(movieName)
        return " "
    next_page_url = urljoin(url, next_page)
    #print("Webpage to be scraped: "+ next_page_url)
    try:
        return movie_description(next_page_url)
    except:
        print(movieName)
        return " "
#movieName=input("Enter Movie Name\n> ")
#print(scrapeDescription(movieName))

pd.options.mode.chained_assignment = None
for x in range(len(movies)):
    movies.iloc[x]["description"]=scrapeDescription(movies.iloc[x]["title"])
    
movies.to_csv('movies_scraped.csv',index=False) 
ratings.to_csv('ratings.csv',index=False)   
userdata.to_csv('users.csv',index=False)


tokenizer = SentenceTransformer('stsb-roberta-large')
movies['embeddings']=movies.description.apply(lambda x: tokenizer.encode(str(x)))
#movies.to_csv('movie_embeddings.csv',index=False)
movies['embeddings'].to_numpy().dump('embeds.pkl')
