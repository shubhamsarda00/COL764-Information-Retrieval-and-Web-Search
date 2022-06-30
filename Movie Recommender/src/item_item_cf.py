# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:52:18 2021

@author: hp
"""

import numpy as np
import pandas as pd
import pickle
import time
from sklearn.model_selection import KFold
#from sentence_transformers import SentenceTransformer
#tokenizer = SentenceTransformer('stsb-roberta-large')

movies=pd.read_csv('data/movies_scraped.csv')
movies.description=movies.description+" "+movies.title+" "+movies.genre
#movies['embeddings']=movies.description.apply(lambda x: tokenizer.encode(str(x)))
movies['embeddings']=pickle.load( open( "data/embeds.pkl", "rb" ) )
ratings=pd.read_csv('data/ratings.csv')

def cos_sim(a,b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def pcc(a,b):
    return sum((a-a.mean())*(b-b.mean()))/(np.sqrt(sum((a-a.mean())**2))*np.sqrt(sum((b-b.mean())**2)))


def recommend(user_id,num_movies):
    i=0
    movie_list=ratings.loc[(ratings['user_id']==user_id  )]
    #movie_list['embeddings']=movie_list.apply(lambda x:tokenizer.encode(x.description))
    user_ratings=ratings.loc[ratings.user_id==user_id]
    preds=pd.DataFrame()
    for x in movies.itertuples():
        i+=1
        print(i)
        if(x.movie_id in movie_list):
            continue
        else:
            emb=x.embeddings
            num=den=0
            for y in user_ratings.itertuples():
                #print(y.movie_id)
                #print(movies.loc[movies.movie_id==y.movie_id].embeddings.item())
                sim=cos_sim(emb,movies.loc[movies.movie_id==y.movie_id].embeddings.item())
                #sim=pcc(emb,movies.loc[movies.movie_id==y.movie_id].embeddings.item())
                num+=y.rating*sim
                den+=sim
            print({'movie_id': x.movie_id,'title':x.title,'rating':num/den})
            temp=pd.DataFrame({'movie_id': [x.movie_id],'title':[x.title],'rating':[num/den]})
            #print(temp)
            preds=preds.append(temp, ignore_index=True)
            #print(preds.iloc[0])
    #scaling
    new_M=user_ratings.rating.max()
    new_m=user_ratings.rating.min()
    m=preds.rating.min()
    M=preds.rating.max()        
    preds.rating=preds.rating.apply(lambda x:(x-m)/(M-m)*(new_M-new_m)+new_m)        
    return preds.sort_values('rating',ascending=False).iloc[:num_movies]        



pd.set_option('mode.chained_assignment', None)
def predict(test,train,k=10):
    preds=test
    preds['rating']=np.nan
    i=0
    for row in preds.itertuples():
        emb=movies.embeddings[row.movie_id==movies.movie_id].item()
        user_ratings=train.loc[train.user_id==row.user_id]
        num=den=0
        temp=[]
       
     
        for y in user_ratings.itertuples():
            sim=cos_sim(emb,movies.loc[movies.movie_id==y.movie_id].embeddings.item())
            #sim2=pcc(emb,movies.loc[movies.movie_id==y.movie_id].embeddings.item())
            #sim=sim1*sim2/(sim1+sim2)
            temp.append([sim,y.rating])
            #num+=y.rating*sim
            #den+=sim
        temp=pd.DataFrame(temp,columns=['similarity','rating'])   

        temp=temp.sort_values('similarity',ascending=False).iloc[:k]
        preds.rating.iloc[i]=(np.dot(temp.rating.values,temp.similarity.values)/temp.similarity.values.sum()-user_ratings.rating.min())/(user_ratings.rating.max()-user_ratings.rating.min())+(user_ratings.rating.min())
        i+=1
        if(i%500==0):
        
            print(i)
    return preds
def trec_eval(data,relevance_threshold,res,qrel) :   
    preds = open('results/'+res+".txt", "w")
    qrels=open('results/'+qrel+'.txt','w')
    a=data.user_id.drop_duplicates().to_numpy()
    user=0
    for u in a:
        user+=1
        #if(user%5000==0):
         #   print(user)
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
        
    
    



kf = KFold(10, shuffle=True,random_state=64)
k=0
for train_in, test_in in kf.split(ratings):
    
    t=time.time()
    k+=1
   
    print(str(k)+" Run")
    ndcg_file = open('results/'+str(k)+'_ndcg_iicf.txt', "w")
    file = open('results/'+str(k)+'_item_item_cf.pkl', 'ab') 
    train=ratings.loc[train_in]
    test=ratings.loc[test_in]
    X_test,y_test=test[['user_id','movie_id']],test.rating
    preds=predict(X_test,train,10)
    predictions = pd.DataFrame({'user_id':X_test.user_id, 'movie_id':X_test.movie_id,'preds':preds.rating.values.flatten(),'truth':test.rating.values.flatten()})    
    trec_eval(predictions,3,'res_iicf_k'+str(k)+'_'+str(3),'qrel_iicf_k'+str(k)+'_'+str(3))
    trec_eval(predictions,4,'res_iicf_k'+str(k)+'_'+str(4),'qrel_iicf_k'+str(k)+'_'+str(4))
    trec_eval(predictions,5,'res_iicf_k'+str(k)+'_'+str(5),'qrel_iicf_k'+str(k)+'_'+str(5))
    
    ndcg_file.write(str(k)+'_fold: Ndcg,Ndcg10,Ndcg100,Ndcg_r,Ndcg10_r,Ndcg100_r are: '+str(NDCG(predictions)))
    ndcg_file.close()
    
    pickle.dump(predictions, file,4)                      
    file.close()     
    print(time.time()-t)



