# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 19:34:10 2022

@author: ASUS
"""

import numpy as np
import pandas as pd
import pickle
import numpy as np
import random 
import math
import time
total=time.time()
from sklearn.model_selection import KFold
movies=pd.read_csv('data/movies_scraped.csv')
movies.description=movies.description+" "+movies.title+" "+movies.genre
#movies['embeddings']=movies.description.apply(lambda x: tokenizer.encode(str(x)))
#movies['embeddings']=pickle.load( open( "embeds.pkl", "rb" ) )
ratings=pd.read_csv('data/ratings.csv')
##############user###########################
len(ratings.movie_id)
pd.set_option('mode.chained_assignment', None)

ratingDict = {}
for row in ratings.itertuples():
    ratingDict[(row.user_id, row.movie_id)] = row.rating


#PEARSON COEFFICIENT to calculate similarity between user_a and user_b
def returnSim(user_a, user_b):

    #find avg ratings given by user1 and user2
    avg_a = ratings.loc[(ratings['user_id']==user_a), 'rating'].mean()
    avg_b = ratings.loc[(ratings['user_id']==user_b), 'rating'].mean()

    sumNum = 0.0 #numerator of the final value in formula
    sumDen1 = 0.0 #1st denominator term
    sumDen2 = 0.0 #2nd denominator term

    for row in movies.itertuples():
        findMovie = row.movie_id

        rating_a = -1
        try:
            rating_a = ratingDict[(user_a, findMovie)]
        except:
            rating_a = -1

        rating_b = -1
        try:
            rating_b = ratingDict[user_b, findMovie]
        except:
            rating_b = -1

        if(not(rating_a == -1 or math.isnan(rating_a))): sumDen1 += pow((rating_a - avg_a),2)
        if(not(rating_b == -1 or math.isnan(rating_b))): sumDen2 += pow((rating_b - avg_b),2)
        if(not(rating_a == -1 or rating_b == -1 or math.isnan(rating_a) or math.isnan(rating_b))):
            sumNum += (rating_a - avg_a)*(rating_b - avg_b)

  
    if(sumDen1 == 0.0 or sumDen2 == 0.0):  
        return 0
    simValue = sumNum/( math.sqrt(sumDen1) * math.sqrt(sumDen2)) 
    return simValue

#Returns similarity of given user to every other user in form of dict 
def simDict_train(userA,train):
    sim2 = {}
    for row in train.itertuples(): 
        if (row.user_id not in sim2) and (row.user_id != userA): #ie iterating over all unique users
            sim2[row.user_id] = returnSim(userA, row.user_id) 
      
    return sim2


#Returns PREDICTED RATING for a given movie by userA

def returnRating(userA, movie, similarity):
    avgA = ratings.loc[(ratings['user_id']==userA), 'rating'].mean()

    sum_Num = 0.0
    sum_Denom = 0.0
    ctr = 0
    k = 100 #how many top users to consider

    for user_i in similarity:
        if(ctr<k):
            ctr += 1

            r_ip = -1 #rating by this top user for that movie
            try:
                r_ip = ratingDict[(user_i, movie)]
            except:
                pass

            if(r_ip == -1 or math.isnan(r_ip)):
                #ctr-=1
                pass 
            else:
                sum_Num += similarity[user_i] * r_ip
                sum_Denom += abs(similarity[user_i])
        else:
            break
  
    predRating = 0
    try:
        predRating = avgA + (sum_Num/sum_Denom)*(1/k)  
    except:
        pass


    return predRating

def trec_eval(data,relevance_threshold, file1path, file2path) :   
    preds = open('results/'+str(file1path) + '.txt', "w")
    qrels=open('results/' + str(file2path) + '.txt','w')
    a=data.user_id.drop_duplicates().to_numpy()
    user=0
    for u in a:
        user+=1
        if(user%5000==0):
            print(user)
        recommendations=data[data['user_id']==u].sort_values('preds',ascending=False)
        
        
        ideal=recommendations.sort_values('truth',ascending=False)
        #if(len(recommendations)==0):
            #   print(u,len(recommendations))
        
        i=0
        #51 0 AP880301-0271 1
        #51 Q0 AP880406-0267 1 0.3434 t2est
        for x in recommendations.itertuples():
            i+=1
            preds.write(str(u)+" Q0 "+str(x.movie_id)+" "+str(i)+" "+str(x.preds)+" t2est"+'\n'+'\n' )
            
        i=0
        for x in ideal.itertuples():
            i+=1
            if(x.truth>=relevance_threshold):
                qrels.write(str(u)+" 0 "+str(x.movie_id)+' '+str(x.truth)+'\n')
            else:
                break

    preds.close()
    qrels.close()
    
    
def NDCG(data):
    a=data.user_id.drop_duplicates().to_numpy()
    
    ndcg=0
    ndcg10=0
    ndcg100=0
    ndcg_r=0
    ndcg10_r=0
    ndcg100_r=0
    user=0
    for u in a:
        user+=1
        if(user%5000==0):
            print(user)
        recommendations=data[data['user_id']==u].sort_values('preds',ascending=False)
        
        
        ideal=recommendations.sort_values('truth',ascending=False)
        #if(len(recommendations)==0):
         #   print(u,len(recommendations))
        dcg=0
        dcg10=0
        dcg100=0
        idcg=0
        idcg10=0
        idcg100=0
        dcg_r=0
        dcg10_r=0
        dcg100_r=0
        idcg_r=0
        idcg10_r=0
        idcg100_r=0
        i=0
        for x in recommendations.itertuples():
            i+=1
            if(i<=10):
                dcg10+=(2**(x.truth)-1)/np.log2(i+1)
                dcg10_r+=(x.truth)/np.log2(i+1)
            if(i<=100):
                dcg100+=(2**(x.truth)-1)/np.log2(i+1)    
                dcg100_r+=x.truth/np.log2(i+1)
            dcg+=(2**(x.truth)-1)/np.log2(i+1)
            dcg_r+=x.truth/np.log2(i+1)
        i=0
        for x in ideal.itertuples():
            i+=1
            if(i<=10):
                idcg10+=(2**(x.truth)-1)/np.log2(i+1)
                idcg10_r+=x.truth/np.log2(i+1)
            if(i<=100):
                idcg100+=(2**(x.truth)-1)/np.log2(i+1)
                idcg100_r+=x.truth/np.log2(i+1)
                
            idcg+=(2**(x.truth)-1)/np.log2(i+1)
            idcg_r+=x.truth/np.log2(i+1) 
        #if(ndcg<2):
         #   print(recommendations.head(10),dcg/idcg)
        ndcg+=dcg/idcg
        ndcg10+=dcg10/idcg10
        ndcg100+=dcg100/idcg100
        ndcg_r+=dcg_r/idcg_r
        ndcg10_r+=dcg10_r/idcg10_r
        ndcg100_r+=dcg100_r/idcg100_r
    ndcg
    return ndcg/len(a),ndcg10/len(a),ndcg100/len(a),ndcg_r/len(a),ndcg10_r/len(a),ndcg100_r/len(a)   


#creating smaller dataset (10k values)

ratings=ratings.sort_values(['user_id']).iloc[0:100000]
ratings=ratings.reset_index(drop=True)

def predict(test,train,k=10):
    preds=test
    preds['rating']=np.nan
    i=0
    #for each row in test set
    for row in preds.itertuples():
        t=time.time()
        similarity = simDict_train(row.user_id,train)
        addVal = returnRating(row.user_id, row.movie_id, similarity)
        preds.rating.iloc[i]= addVal
        #print(addVal)
        #print(preds.rating.iloc[i])
        i+=1
        if(i==1):
            print(time.time()-t)
        if(i%500==0):
            print(i)
    return preds

kf = KFold(10,shuffle=True,random_state=64)
k=0
for train_in, test_in in kf.split(ratings):
    
    t=time.time()
    k+=1
    #if(k==10):
    print(str(k)+" Run")
    ndcg_file = open('results/'+str(k)+'_ndcg_uucf.txt', "w")
    file1 = open('results/'+str(k)+'_user_user_cf.pkl', 'wb') 
    train=ratings.loc[train_in]
    test=ratings.loc[test_in]
    X_test,y_test=test[['user_id','movie_id']],test.rating

    preds=predict(X_test,train,10)
    predictions = pd.DataFrame({'user_id':X_test.user_id, 'movie_id':X_test.movie_id,'preds':preds.rating.values.flatten(),'truth':test.rating.values.flatten()})    
    trec_eval(predictions,3,'res_uucf_k'+str(k)+'_'+str(3),'qrel_uucf_k'+str(k)+'_'+str(3))
    trec_eval(predictions,4,'res_uucf_k'+str(k)+'_'+str(4),'qrel_uucf_k'+str(k)+'_'+str(4))
    trec_eval(predictions,5,'res_uucf_k'+str(k)+'_'+str(5),'qrel_uucf_k'+str(k)+'_'+str(5))

    ndcg_file.write(str(k)+'_fold: Ndcg,Ndcg10,Ndcg100,Ndcg_r,Ndcg10_r,Ndcg100_r are: '+str(NDCG(predictions)))
    ndcg_file.close()

    pickle.dump(predictions, file1,4)                      
    file1.close()     
    print(time.time()-t)
    
    
