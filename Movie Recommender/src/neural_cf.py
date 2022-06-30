# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 20:00:51 2021

@author: hp
"""

import numpy as np
import tensorflow as tf
import random 

#np.random.seed(14)
#random.seed(14)
#tf.random.set_seed(14)
from sklearn.preprocessing import LabelEncoder
import pandas as pd

import keras
import pickle
import time
total=time.time()
from keras.layers import Embedding
from keras.models import Model
from keras.layers import *
from keras_self_attention import SeqSelfAttention
from keras.callbacks import ModelCheckpoint, EarlyStopping,LearningRateScheduler
from keras.initializers import TruncatedNormal
import sys
model_num=sys.argv[1]

movies=pd.read_csv('data/movies_scraped.csv')
movies.description=movies.description+" "+movies.title+" "+movies.genre
#movies['embeddings']=movies.description.apply(lambda x: tokenizer.encode(str(x)))
movies['embeddings']=pickle.load( open( "data/embeds.pkl", "rb" ) )
ratings=pd.read_csv('data/ratings.csv')

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
        

filtered_movies=movies[movies.movie_id.isin(ratings.movie_id.drop_duplicates()) ]
filtered_movies.movie_id.nunique(),ratings.movie_id.nunique()
w=np.stack(filtered_movies.embeddings.values)
# Ordinal encoding
user_enc = LabelEncoder()
#small_dataset.user_id = user_enc.fit_transform(small_dataset.user_id.values)
ratings.user_id = user_enc.fit_transform(ratings.user_id.values)
movie_enc = LabelEncoder()
#small_dataset.movie_id = movie_enc.fit_transform(small_dataset.movie_id.values)
ratings.movie_id = movie_enc.fit_transform(ratings.movie_id.values)
filtered_movies.movie_id=movie_enc.transform(filtered_movies.movie_id)
#filtered_movies.movie_id=movie_enc.transform(filtered_movies.movie_id)

#############################################################################
###############################MODELS########################################
 
def NCF_Model1(n_features,n_items,n_users):
    
    U=Input(shape=(1,), name='User')
    M=Input(shape=(1,),name='Movie')
    path1=Embedding(n_users,n_features)(U)
    path2=Embedding(n_items,n_features,name='Movie_Embedding')(M)
    ####for intializing with sentence embeddings
    #embedding.set_weights([pretrained_embeddings])
    path1=Reshape((n_features,))(path1)
    path2=Reshape((n_features,))(path2)
    merged=Dot(axes=1)([path1,path2])
    
    #merged= Dense(32,activation='relu', kernel_initializer='uniform')(merged)
    #merged = BatchNormalization()(merged)
    #merged= Dense(64,activation='relu', kernel_initializer='uniform')(merged)
    #merged = BatchNormalization()(merged)
    #merged= Dense(128,activation='relu', kernel_initializer='uniform')(merged)
    #merged = BatchNormalization()(merged)
    #merged= Dense(256,activation='relu', kernel_initializer='uniform')(merged)
    #merged = BatchNormalization()(merged)
    #merged= Dense(128,activation='relu', kernel_initializer='uniform')(merged)
    #merged = BatchNormalization()(merged)
    #merged= Dense(64,activation='relu', kernel_initializer='he_normal')(merged)
    #merged = BatchNormalization()(merged)
    merged= Dense(32,activation='relu', kernel_initializer='uniform')(merged)
    #merged = BatchNormalization()(merged)
    merged= Dense(16,activation='relu', kernel_initializer='uniform')(merged)
    #merged = BatchNormalization()(merged)
    merged= Dense(8,activation='relu', kernel_initializer='uniform')(merged)
    #merged = BatchNormalization()(merged)
    merged= Dense(1,activation='sigmoid', kernel_initializer='lecun_uniform')(merged)
    merged=Lambda(lambda x:x*5)(merged)
    
    model = Model(inputs=[U,M], outputs=merged, name='NCF_Model_1')
    return model

def NCF_Model2(n_features,n_items,n_users):
    
    U=Input(shape=(1,), name='User')
    M=Input(shape=(1,),name='Movie')
    
    path1=Embedding(n_users,n_features,name='User_Embedding')(U)
    
    
    path2=Embedding(n_items,n_features,name='Pre-trained_Movie_Embedding')(M)
    
    ####for intializing with sentence embeddings
    #embedding.set_weights([pretrained_embeddings])
    path1=Reshape((n_features,))(path1)
    path1=Dense(256,activation='gelu', kernel_initializer='uniform')(path1)
    path1 = BatchNormalization()(path1)
    path2=Reshape((n_features,))(path2)
    path2=Dense(256,activation='gelu', kernel_initializer='uniform')(path2)
    path2 = BatchNormalization()(path2)
    #merged=(axes=1)([path1,path2])
    #merged = Add()([path1, path2])
    merged = Concatenate()([path1, path2])
    merged= Dense(512,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(256,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged=Dropout(0.1)(merged)
    merged= Dense(128,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(64,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged=Dropout(0.1)(merged)
    merged= Dense(32,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(16,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(8,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(1,activation='gelu', kernel_initializer='uniform')(merged)
    
    
    model = Model(inputs=[U,M], outputs=merged, name='NCF_Model_2')
    return model


def NCF_Model3(n_features,n_items,n_users):
    
    U=Input(shape=(1,), name='User')
    M=Input(shape=(1,),name='Movie')
    
    path1=Embedding(n_users,n_features, name='User_Embedding')(U)
    
    
    path2=Embedding(n_items,n_features,name='Pre-trained_Movie_Embedding')(M)
    
    ####for intializing with sentence embeddings
    #path1.set_weights([a])
    path1=Reshape((n_features,))(path1)
    path1=Dense(128,activation='gelu', kernel_initializer='uniform')(path1)
    
    path2=Reshape((n_features,))(path2)
   

    path2=Dense(128,activation='gelu', kernel_initializer='uniform')(path2)
    #merged=(axes=1)([path1,path2])
    merged = Concatenate()([path1, path2])
    #merged=Add()([path1,path2])
    
    merged= Dense(128,activation='gelu', kernel_initializer='uniform')(merged)
    merged_1 = BatchNormalization()(merged)
    merged= Dense(256,activation='gelu', kernel_initializer='uniform')(merged_1)
    merged_2 = BatchNormalization()(merged)
    merged=Dropout(0.1)(merged_2)
    merged= Dense(512,activation='gelu', kernel_initializer='uniform')(merged)
    merged_3 = BatchNormalization()(merged)
    merged= Dense(1024,activation='gelu', kernel_initializer='uniform')(merged_3)
    merged = BatchNormalization()(merged)
    merged= Dense(512,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    ###########Skip connection 1
    merged= Add()([merged, merged_3])
    
    merged= Dense(256,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    ###########Skip connection 2
    merged= Add()([merged, merged_2])
    
    
    merged= Dense(128,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    ###########Skip connection 3
    merged=Add()([merged_1,merged])
    
    merged= Dense(64,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged=Dropout(0.1)(merged)
    merged= Dense(32,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(16,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(8,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(1,activation='gelu', kernel_initializer='uniform')(merged)
    
    
    
    model = Model(inputs=[U,M], outputs=merged, name='NCF_Model_3')
    return model
def NCF_Model6(n_features,n_items,n_users):
    
    U=Input(shape=(1,), name='User')
    M=Input(shape=(1,),name='Movie')
    
    path1=Embedding(n_users,n_features)(U)
    
    
    path2=Embedding(n_items,n_features,name='Pre-trained_Movie_Embedding')(M)
    
    ####for intializing with sentence embeddings
    #path1.set_weights([a])
    path1=Reshape((n_features,))(path1)
    path1=Dense(256,activation='gelu', kernel_initializer='uniform')(path1)
    
    #path2=Reshape((n_features,))(path2)
    path2=Bidirectional(LSTM(256,return_sequences=True))(path2)
    path2=SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,attention_activation='sigmoid')(path2)
    path2=TimeDistributed(Dense(8,activation='gelu', kernel_initializer='uniform'))(path2)
    path2=Bidirectional(LSTM(256))(path2)
    #path2=SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,attention_activation='sigmoid')(path2)
    path2=Dense(256,activation='gelu', kernel_initializer='uniform')(path2)
    #merged=(axes=1)([path1,path2])
    merged = Concatenate()([path1, path2])
    merged= Dense(512,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(256,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged=Dropout(0.1)(merged)
    merged= Dense(128,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged=Dropout(0.1)(merged)
    merged= Dense(64,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(32,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(16,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(8,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(1,activation='gelu', kernel_initializer='uniform')(merged)
    #merged=Lambda(lambda x:x*4+1)(merged)
    
    model = Model(inputs=[U,M], outputs=merged, name='NCF_Model_4')
    return model
 
 
def NCF_Model7(n_features,n_items,n_users):
    
    U=Input(shape=(1,), name='User')
    M=Input(shape=(1,),name='Movie')
    
    path1=Embedding(n_users,n_features)(U)
    
    
    path2=Embedding(n_items,n_features,name='Pre-trained_Movie_Embedding')(M)
    
    ####for intializing with sentence embeddings
    #path1.set_weights([a])
    path1=Reshape((n_features,))(path1)
    path1=Dense(256,activation='gelu', kernel_initializer='uniform')(path1)
    
    #path2=Reshape((n_features,))(path2)
    path2=Bidirectional(LSTM(256,return_sequences=True))(path2)
    path2=SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,attention_activation='sigmoid')(path2)
    path2=TimeDistributed(Dense(8,activation='gelu', kernel_initializer='uniform'))(path2)
    path2=Bidirectional(LSTM(256))(path2)
    #path2=SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,attention_activation='sigmoid')(path2)
    path2=Dense(256,activation='gelu', kernel_initializer='uniform')(path2)
    #merged=(axes=1)([path1,path2])
    merged = Concatenate()([path1, path2]) 
    
    merged= Dense(128,activation='gelu', kernel_initializer='uniform')(merged)
    merged_1 = BatchNormalization()(merged)
    merged= Dense(256,activation='gelu', kernel_initializer='uniform')(merged_1)
    merged_2 = BatchNormalization()(merged)
    merged=Dropout(0.1)(merged_2)
    merged= Dense(512,activation='gelu', kernel_initializer='uniform')(merged)
    merged_3 = BatchNormalization()(merged)
    merged= Dense(1024,activation='gelu', kernel_initializer='uniform')(merged_3)
    merged = BatchNormalization()(merged)
    merged= Dense(512,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    ###########Skip connection 1
    merged= Add()([merged, merged_3])
    
    merged= Dense(256,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    ###########Skip connection 2
    merged= Add()([merged, merged_2])
    
    
    merged= Dense(128,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    ###########Skip connection 3
    merged=Add()([merged_1,merged])
    
    merged= Dense(64,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged=Dropout(0.1)(merged)
    merged= Dense(32,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(16,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(8,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(1,activation='gelu', kernel_initializer='uniform')(merged)
    
    
    
    model = Model(inputs=[U,M], outputs=merged, name='NCF_Model_5')
    return model
    
def NCF_Model4(n_features,n_items,n_users):
    
    U=Input(shape=(1,), name='User')
    M=Input(shape=(1,),name='Movie')
    
    path1=Embedding(n_users,n_features)(U)
    
    
    path2=Embedding(n_items,(40*300),name='Pre-trained_Movie_Embedding',)(M)
    
    ####for intializing with sentence embeddings
    #path1.set_weights([a])
    path1=Reshape((n_features,))(path1)
    path1=Dense(256,activation='gelu', kernel_initializer='uniform')(path1)
    
    path2=Reshape((40,300))(path2)
    path2=Bidirectional(LSTM(256,input_shape=(40,300),return_sequences=True))(path2)
    path2=SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,attention_activation='sigmoid')(path2)
    path2=TimeDistributed(Dense(8,activation='gelu', kernel_initializer='uniform'))(path2)
    path2=Bidirectional(LSTM(256))(path2)
    #path2=SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,attention_activation='sigmoid')(path2)
    path2=Dense(256,activation='gelu', kernel_initializer='uniform')(path2)
    #merged=(axes=1)([path1,path2])
    merged = Concatenate()([path1, path2])
    merged= Dense(512,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(256,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged=Dropout(0.1)(merged)
    merged= Dense(128,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(64,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged=Dropout(0.1)(merged)
    merged= Dense(32,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(16,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(8,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(1,activation='gelu', kernel_initializer='uniform')(merged)
    
    
    model = Model(inputs=[U,M], outputs=merged, name='NCF_Model_6')
    return model
    
def NCF_Model5(n_features,n_items,n_users):
    
    U=Input(shape=(1,), name='User')
    M=Input(shape=(1,),name='Movie')
    
    path1=Embedding(n_users,n_features)(U)
    
    
    path2=Embedding(n_items,(40*300),name='Pre-trained_Movie_Embedding',)(M)
    
    ####for intializing with sentence embeddings
    #path1.set_weights([a])
    path1=Reshape((n_features,))(path1)
    path1=Dense(256,activation='gelu', kernel_initializer='uniform')(path1)
    
    path2=Reshape((40,300))(path2)
    path2=Bidirectional(LSTM(256,input_shape=(40,300),return_sequences=True))(path2)
    path2=SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,attention_activation='sigmoid')(path2)
    path2=TimeDistributed(Dense(8,activation='gelu', kernel_initializer='uniform'))(path2)
    path2=Bidirectional(LSTM(256))(path2)
    #path2=SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,attention_activation='sigmoid')(path2)
    path2=Dense(256,activation='gelu', kernel_initializer='uniform')(path2)
    #merged=(axes=1)([path1,path2])
    merged = Concatenate()([path1, path2]) 
    
    merged= Dense(128,activation='gelu', kernel_initializer='uniform')(merged)
    merged_1 = BatchNormalization()(merged)
    merged= Dense(256,activation='gelu', kernel_initializer='uniform')(merged_1)
    merged_2 = BatchNormalization()(merged)
    merged=Dropout(0.1)(merged_2)
    merged= Dense(512,activation='gelu', kernel_initializer='uniform')(merged)
    merged_3 = BatchNormalization()(merged)
    merged= Dense(1024,activation='gelu', kernel_initializer='uniform')(merged_3)
    merged = BatchNormalization()(merged)
    merged= Dense(512,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    ###########Skip connection 1
    merged= Add()([merged, merged_3])
    
    merged= Dense(256,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    ###########Skip connection 2
    merged= Add()([merged, merged_2])
    
    
    merged= Dense(128,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    ###########Skip connection 3
    merged=Add()([merged_1,merged])
    
    merged= Dense(64,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged=Dropout(0.1)(merged)
    merged= Dense(32,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(16,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(8,activation='gelu', kernel_initializer='uniform')(merged)
    merged = BatchNormalization()(merged)
    merged= Dense(1,activation='gelu', kernel_initializer='uniform')(merged)
    
    
    
    model = Model(inputs=[U,M], outputs=merged, name='NCF_Model_7')
    return model
################################################################################################################################
################################################################################################################################
    
def NCF_Model(num):
    
    if(num=='1'):
        m=NCF_Model1(1024,3706,6040)
       
    elif(num=='2'):
        m=NCF_Model2(1024,3706,6040)
        m.get_layer('Pre-trained_Movie_Embedding').set_weights([w])
       
    elif(num=='3'):
        m=NCF_Model3(1024,3706,6040)
        m.get_layer('Pre-trained_Movie_Embedding').set_weights([w])
    elif(num=='6'):
        m=NCF_Model6(1024,3706,6040)  
        m.get_layer('Pre-trained_Movie_Embedding').set_weights([w])
    elif(num=='7'):
        m=NCF_Model7(1024,3706,6040)  
        m.get_layer('Pre-trained_Movie_Embedding').set_weights([w]) 
    elif(num=='4'):
        m=NCF_Model4(300,3706,6040) 
        import spacy
        import en_core_web_lg
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        s=set(stopwords.words('english'))
        word_encoder=en_core_web_lg.load() 
        new_w=np.zeros((3706,40,300))
        word_encoder.max_length
        i=0
        l=0
        for x in filtered_movies.itertuples():
            p=0
            i+=1
            if(i%500==0):
             print(i)
            temp=[]
            desc=str(x.description).lower()
            punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~0123456789'''
        
            
            new = ""
            for char in desc:
                if char not in punc:
                    new = new + char
        
            new=' '.join(new.split())
            
            
            for word in word_encoder(new):
                if word.text in s:
                    continue
                if(i==1):
                    print (word.text)
        
                p+=1
                temp.append(word.vector.astype('float32'))
            while(len(temp)<40):
                temp.append(np.zeros(300))
            l=max(l,p)
            new_w[int(x.movie_id)]=np.array(temp).astype('float32')
        new_w=new_w.reshape(3706,40*300) 
         
        
        m.get_layer('Pre-trained_Movie_Embedding').set_weights([new_w])     
        
        
    elif(num=='5'):
        m=NCF_Model5(300,3706,6040)  
        import spacy
        import en_core_web_lg
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        s=set(stopwords.words('english'))
        word_encoder=en_core_web_lg.load() 
        new_w=np.zeros((3706,40,300))
        word_encoder.max_length
        i=0
        l=0
        for x in filtered_movies.itertuples():
            p=0
            i+=1
            if(i%500==0):
             print(i)
            temp=[]
            desc=str(x.description).lower()
            punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~0123456789'''
        
            
            new = ""
            for char in desc:
                if char not in punc:
                    new = new + char
        
            new=' '.join(new.split())
            
            
            for word in word_encoder(new):
                if word.text in s:
                    continue
                if(i==1):
                    print (word.text)
        
                p+=1
                temp.append(word.vector.astype('float32'))
            while(len(temp)<40):
                temp.append(np.zeros(300))
            l=max(l,p)
            new_w[int(x.movie_id)]=np.array(temp).astype('float32')
        new_w=new_w.reshape(3706,40*300) 
        
        m.get_layer('Pre-trained_Movie_Embedding').set_weights([new_w])   
        
    m.compile(loss='mse', optimizer=keras.optimizers.Adamax(learning_rate=0.0005))
    mc = ModelCheckpoint('results/NCF'+str(model_num)+'.h5', monitor='val_loss', mode='min', verbose=1,save_best_only=True,save_weights_only=True)
    es=EarlyStopping(monitor="val_loss",min_delta=0,patience=10,verbose=1,mode="min",restore_best_weights=True)
    def lrs(epoch,lr):
        if epoch>1:
            return 0.001
        else:
            return lr
    lr_sched=LearningRateScheduler(lrs)    
    #cb=[mc,es,lr_sched]    
    cb=[mc,es]
    return cb,m

from sklearn.model_selection import KFold
kf = KFold(10,shuffle=True,random_state=64)
k=0
for train_in, test_in in kf.split(ratings):
    
    t=time.time()
    k+=1
    
    print(str(k)+" Run")
#    if not(k>7):
#        continue
    ndcg_file = open('results/'+str(k)+'split_ndcg_ncf'+str(model_num)+'.txt', "w")
    #file = open('COL764-Project/'+str(k)+'split_ncf'+str(model_num)+'.pkl', 'ab') 
    train=ratings.loc[train_in]
    test=ratings.loc[test_in]
    X_train,y_train=train[['user_id','movie_id']].to_numpy(),train.rating.to_numpy()
    X_test,y_test=test[['user_id','movie_id']].to_numpy(),test.rating.to_numpy()
    X_train = [X_train[:, 0], X_train[:, 1]]
    X_test = [X_test[:, 0], X_test[:, 1]]
    
    cb,m=NCF_Model(model_num)
    
    #m.compile(loss='mse', optimizer=keras.optimizers.Adamax(learning_rate=0.0003))
    
    m.fit(X_train,y_train,batch_size=256,epochs=5000,validation_data=(X_test,y_test),callbacks=cb)
    y_pred=m.predict(X_test)
    predictions = pd.DataFrame({'user_id':X_test[0], 'movie_id':X_test[1],'preds':y_pred.flatten(),'truth':y_test})
    predictions.to_csv('results/'+str(k)+'split_ncf'+str(model_num)+'.csv')
    trec_eval(predictions,3,'res_ncf'+str(model_num)+'_k'+str(k)+'_'+str(3),'qrel_ncf'+str(model_num)+'_k'+str(k)+'_'+str(3))
    trec_eval(predictions,4,'res_ncf'+str(model_num)+'_k'+str(k)+'_'+str(4),'qrel_ncf'+str(model_num)+'_k'+str(k)+'_'+str(4))
    trec_eval(predictions,5,'res_ncf'+str(model_num)+'_k'+str(k)+'_'+str(5),'qrel_ncf'+str(model_num)+'_k'+str(k)+'_'+str(5))
    
    ndcg_file.write(str(k)+'_fold: Ndcg,Ndcg10,Ndcg100,Ndcg_r,Ndcg10_r,Ndcg100_r are: '+str(NDCG(predictions)))
    ndcg_file.close()
    #pickle.dump(predictions, file,4)                      
    #file.close()     
    print(time.time()-t)
print(time.time()-total)    
    

