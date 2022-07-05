# Movie Recommender: Course Project

This project implements a **Movie Recommendation System** for users in the **MovieLens 1M database** (https://grouplens.org/datasets/movielens/1m/). 

### Dataset

The dataset contains 1,000,000 ratings given by different users along with the timestamp of viewing, as well as basic details about the movie (title, genres). We further process the **.dat** files from the original dataset to create **.csv** files for ease of use. In order to make our dataset more comprehensive, we have performed **web scraping** from **IMDb** over all the movies of the MovieLens 1M dataset, and augmented the MovieLens dataset with the movie descriptions obtained from there. We have extracted the movie description from the first result displayed in IMDb search for each movie in the MovieLens dataset. We have not used additional search results since IMDb only contains one unique correct result corresponding to a movie title, and using lower ranked search results had a higher chance of appending incorrect movie descriptions to our dataset.

Movies dataframe before webscraping:

![image](https://user-images.githubusercontent.com/45795080/177042966-d3ca32f0-b093-4f28-b4b6-1dd16305a9f7.png)

Movies dataframe after webscraping:

![image](https://user-images.githubusercontent.com/45795080/177042953-162b7276-04bc-4e53-b3d2-ca57961e735f.png)


### Our Recommendation System

![image](https://user-images.githubusercontent.com/45795080/177255843-fb726b97-6b64-4b02-bb3e-e6a79ae20e5f.png)


1. We have implemented multiple (9) models to perform a **“Recommended for You”** functionality for a user based on their previous ratings.

a) **User-User Collaborative Filtering**: 
<br/>
b) **Item-Item Collaborative Filtering**: 
<br/>
c) **Neural Collaborative Filtering**:
<br/>

2. We have implemented 1 model for **“Because you watched [Movie X recently, you may also like]”** option based on the last watched movie by a user (inspired by the similar option provided by **Netflix**).

3. We have implemented 3 models for a new user (**cold start case**), whose previous data is not available to us:
 
a) Recommending the most recent top-rated movies <br/>
b) Recommending movies similar to their favourite movie <br/>
c) Recommending top-rated movies of a genre of their choice <br/><br/>

**More details in "Movie_Recommender_Report.pdf" and "Movie Recommender System - Team Algo Busters.pptx".**

### Evaluation and Results

We have run 10-fold cross validation on each of our models, wherein we evaluated the models 10 times by dividing the dataset into train and test sets in a 90:10 ratio each time. For item-item collaborative filtering and our models for neural collaborative filtering, we have used the entire 1M dataset for our 10-fold cross-validation. For user-user collaborative filtering, due to the higher time complexity with respect to unwatched movies, we have run the cross-validation on a test set of 1/10th size.

We have primarily used the following metrics in our evaluation:
1) RMSE (Root Mean Squared Error)
2) NDCG, NDCG@10
3) Recall@10, Recall@100
4) Precision@10, Precision@100
5) MRR (Mean Reciprocal Rank)
6) MAP (Mean Average Precision)
7) Statistical Significance Tests: Paired Student’s t-test and Wilcoxon Signed Rank Test

We've used **Text REtrieval Conference (TREC)** evaluation software (https://github.com/usnistgov/trec_eval) to calculate the said metrics.
We’ve created Qrel files by sorting the test set using ground truth ratings. We’ve experimented with three threshold (user) ratings- 3, 4 and 5 to consider a movie relevant. For creating the results file, we sorted the test set on the basis of predicted ratings. Hence, for each fold of cross-validation, we have generated 3 corresponding qrels files as well as 3 possible results files in order to run these files on trec_eval and calculate the relevant metrics. Complete results can be found in:<br/> https://docs.google.com/spreadsheets/d/1KVXD6MGQhXYBByOostWGwVohUvSFoYZ9a-9GHZVUo3o/edit#gid=0
 
We find that on almost all metrics, our best results are obtained on our 6th and 7th Neural Collaborative Filtering architectures. We have not included user-user collaborative filtering results in our comparison since that was performed on a smaller dataset. In terms of RMSE, current state-of-the-art model for this dataset obtains a rmse score of **0.822** compared to **0.866** achieved by our best model of Neural Collaborative Filtering. On comparing, our rmse results for several neural architecture are very close to the latest sota results, For NDCG, although they haven't specified the relevance threshold used for NDCG calculation, our neural collaborative filtering models are in line with the current sota results (https://paperswithcode.com/sota/collaborative-filtering-on-movielens-1m). 



### Running the Code

1. **Movie_Recommender.ipynb**:<br/>

2. **preprocessing.py**:<br/>

3. **user_user_cf.py**:<br/>

4. **item_item_cf.py**:<br/>

5. **neural_cf.py**:<br/>

### Libraries Required

1)NLTK - https://www.nltk.org/ 

2)spaCy - https://spacy.io/ 

3)NumPy - https://numpy.org/ 

4)Pandas - https://pandas.pydata.org/ 

5)sentence-transformers - https://github.com/UKPLab/sentence-transformers

6)Keras - https://keras.io/ 

7)Keras Self Attention - https://github.com/CyberZHG/keras-self-attention

8)TensorFlow - https://www.tensorflow.org/ 

9)scikit-learn - https://scikit-learn.org/stable/ 
