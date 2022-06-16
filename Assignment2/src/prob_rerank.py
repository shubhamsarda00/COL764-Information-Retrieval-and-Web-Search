# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:16:07 2020

@author: hp
"""

import sys,math
#import pickle,bz2
from nltk.tokenize import word_tokenize as tokenizer
#tryspacy
from nltk.corpus import stopwords
#nltk.download("stopwords")
#nltk.download("punkt")
from krovetz import PyKrovetzStemmer
import time

docfile=sys.argv[3]
m=int(sys.argv[4])
top100file=sys.argv[2]
queryfile=sys.argv[1]

#tokenizer("nkn")
topdocs={}
#t=time.time()
f=open(top100file, 'r')
doclist={}
while(True):
    
    temp=f.readline().split()
    if(len(temp)==0):
        break
    if(temp[0] not in topdocs):
        topdocs[temp[0]]={}

    topdocs[temp[0]][temp[2]]={"score":temp[4]}
    doclist[temp[2]]=1
f.close()
#t1=time.time()
#print(t1-t)

#print(len(doclist))
f=open(docfile, 'r')
i=0
while(True):
    i+=1
    temp=f.readline()[:-1].split("\t")
    #print(temp)
    
    
    if(len(temp)<4):
        
        break
    if(temp[0] in doclist):
        
        #print(temp[0])
        
        doclist[temp[0]]=temp[3].lower()
  
sw = set(stopwords.words('english')) 
#sw=set({'during', 're', 'mustn', 'wouldn', "wouldn't", 'now', "wasn't", 'how', 'all', 'our', 'herself', 'against', 'couldn', 'in', "mustn't", "you've", 'should', 'under', 'won', 'and', 'having', 'been', 'or', 'just', 'mightn', 'its', 'm', 'own', 'there', 'below', 'while', 'do', 'of', 'after', "didn't", 'him', 'those', 'whom', 'only', 'again', 'hasn', 'himself', 'are', 'wasn', 'that', 'doing', 'shan', 'it', 'y', 's', "don't", "weren't", 'hadn', 'my', 'yours', 'very', 'up', 'have', 'these', 'between', "you'd", 'until', 'for', 'them', 'then', 'myself', 'is', "shouldn't", "you're", "she's", 'before', 'such', 'no', 'off', 'she', 'we', 'ma', 'this', 'yourselves', 'his', 'an', 'ourselves', 'but', 'above', "it's", 'doesn', 'through', 'here', 'shouldn', 'when', 'out', 'does', 've', 'once', "isn't", 'am', 'being', 't', 'at', 'hers', 'same', "couldn't", 'few', "shan't", 'their', 'most', 'll', "won't", 'which', 'with', 'why', 'o', 'more', 'than', 'isn', 'did', 'further', 'other', 'didn', 'as', 'if', 'me', 'to', 'on', 'into', 'don', "doesn't", 'from', 'some', "you'll", 'd', 'itself', 'will', 'haven', 'be', "hasn't", 'about', 'your', "hadn't", 'aren', "that'll", 'has', 'any', 'weren', 'needn', "haven't", 'so', 'were', "aren't", 'what', 'a', 'by', 'not', 'had', 'nor', 'they', 'theirs', 'ain', 'i', 'can', 'you', "mightn't", 'over', 'because', 'he', 'themselves', 'was', 'each', 'too', "needn't", 'ours', 'where', 'down', 'her', 'the', 'both', "should've", 'yourself', 'who'})
krovetzstemmer=PyKrovetzStemmer()
#tokenizer=tokenizer()
N=3213835

k1=1.7
b=0.78


f=open(queryfile, 'r')
queries={}
while True :
    temp=f.readline()[:-1].split("\t")
    if(len(temp)==1):
        break
    
    q=temp[0]
    #print(temp)
    tokens=tokenizer(temp[1].lower())
    #tokens=temp[1].lower().split()
    tokens=[word for word in tokens if word not in sw]
    tokens=list(map(krovetzstemmer.stem,tokens))
    
    queries[q]=tokens


'''
s=bz2.BZ2File('indexfile'+'.dict','w')
pickle.dump(topdocs,s,protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(queries,s,protocol=pickle.HIGHEST_PROTOCOL)    
pickle.dump(doclist,s,protocol=pickle.HIGHEST_PROTOCOL)

s = bz2.BZ2File("indexfile.dict", 'rb')
topdocs=pickle.load(s)
queries=pickle.load(s)
doclist=pickle.load(s)
'''
Finalresult={}


j=0
for q,val in topdocs.items():
    ###########indexing
    #t=time.time()
    #j+=1
 
    index={}    
    dlavg=0
    for docid in val:
        tokens=tokenizer(doclist[docid])
        tokens=[word for word in tokens if word not in sw]
        tokens=list(map(krovetzstemmer.stem,tokens))
        topdocs[q][docid]["length"]=len(tokens)
        dlavg+=len(tokens)
        for word in tokens:
            if(len(word)==1):
                continue
            if word in index:
                index[word][docid]=index[word].get(docid,0)+1
            else:
                index[word]={docid:1}
   
    dlavg/=len(topdocs[q])
    topdocs[q]["dlavg"]=dlavg
    ######### term selection and bm25 scoring
    top={}
    
    for word,posting in index.items():
        #df=len(k)-1 # RANKS shouldn't be counted
        df=len(posting)
        DF=df/100*N
        w=round(math.log((N+2*DF)/DF),4)
        top[word]=round(w*df,4)       #  VRi*wi
        dlavg=topdocs[q]["dlavg"]
        '''
        for docid,tf in posting.items():
            dl=topdocs[q][docid]["length"]
            posting[docid]=round(w*(tf*(1+k1))/(k1*(1-b+b*(dl/dlavg))+tf),4)
        '''    
    top=sorted(top.items(),key=lambda x: -x[1])[:m]
    
    
    ##############################Query processing
    tokens=queries[q]
    score={}
    
    for word in tokens:
        if(len(word)==1):
            continue
        if(word not in index):
            continue
        posting=index[word]
        df=len(posting)
        DF=df/100*N
        w=round(math.log((N+2*DF)/DF),4)
        for docid in posting:
            dl=topdocs[q][docid]["length"]
            tf=posting[docid]
            #score[docid]=score.get(docid,0)+posting[docid]
            score[docid]=score.get(docid,0)+round(w*(tf*(1+k1))/(k1*(1-b+b*(dl/dlavg))+tf),4)
    #Fullscore[q] = max(Fullscore[q].items(), key= lambda x: x[1])
    #Fullscore[q]=sorted(Fullscore[q].items(),key= lambda x: -x[1])
    result=[]
    for i in range(m):
        posting=index[top[i][0]]
        df=len(posting)
        DF=df/100*N
        w=round(math.log((N+2*DF)/DF),4)
        for docid in posting:
            dl=topdocs[q][docid]["length"]
            tf=posting[docid]
            score[docid]=score.get(docid,0)+round(w*(tf*(1+k1))/(k1*(1-b+b*(dl/dlavg))+tf),4)
        result.append(max(score.items(), key= lambda x: x[1]))
    Finalresult[q]=result
    #print(time.time()-t,j)

tem=sorted(list(Finalresult.keys()))    
for i in range(m):
    f = open("result_pb"+str(i+1)+".txt", "w")
    for q in tem:
        
        a=Finalresult[q][i]
        f.write(str(q)+" Q0 "+a[0]+" "+str(1)+" "+str(a[1])+" t2est"+'\n' )
    f.close()
  
  
    