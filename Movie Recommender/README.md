# Movie Recommender: Course Project

This project implements a movie recommendation system for users in the **MovieLens 1M database** (https://grouplens.org/datasets/movielens/1m/). 

### Dataset

The dataset contains 1,000,000 ratings given by different users along with the timestamp of viewing, as well as basic details about the movie (title, genres). We further process the **.dat** files from the original dataset to create **.csv** files for ease of use. In order to make our dataset more comprehensive, we have performed **web scraping** from **IMDb** over all the movies of the MovieLens 1M dataset, and augmented the MovieLens dataset with the movie descriptions obtained from there. We have extracted the movie description from the first result displayed in IMDb search for each movie in the MovieLens dataset. We have not used additional search results since IMDb only contains one unique correct result corresponding to a movie title, and using lower ranked search results had a higher chance of appending incorrect movie descriptions to our dataset.

Movies dataframe before webscraping:

![image](https://user-images.githubusercontent.com/45795080/177042966-d3ca32f0-b093-4f28-b4b6-1dd16305a9f7.png)

Movies dataframe after webscraping:

![image](https://user-images.githubusercontent.com/45795080/177042953-162b7276-04bc-4e53-b3d2-ca57961e735f.png)


### Our Recommendation System

We provide recommendations based on the user’s overall past
preferences, based on their most recently watched films, as
well as some basic recommendations in case of a new user
who hasn’t previously used Movie Lens. We have employed
item-item collaborative filtering, user-user collaborative
filtering, and multiple separate neural architectures that
implement neural collaborative filtering. Further, we
have performed 10-fold cross-validation on each of our
architectures in order to evaluate their performance
