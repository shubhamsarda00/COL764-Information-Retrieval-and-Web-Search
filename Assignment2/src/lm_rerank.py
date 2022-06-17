# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:16:07 2020

@author: hp
"""

import sys,math

from nltk.tokenize import word_tokenize as tokenizer

#tryspacy
from nltk.corpus import stopwords
#nltk.download("stopwords")
#nltk.download("punkt")
from krovetz import PyKrovetzStemmer
import time

docfile=sys.argv[3]
model=sys.argv[4]
top100file=sys.argv[2]
queryfile=sys.argv[1]


topdocs={}

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


#print(len(doclist))
f=open(docfile, 'r')
i=0
while(True):
    i+=1
    temp=f.readline()[:-1].split("\t")
    #print(temp)
    
    #if(i%100000==0):
     #   print(i)
    if(len(temp)<4):
        #print(i)
        break
    if(temp[0] in doclist):
        
        #print(temp[0])
        
        doclist[temp[0]]=temp[3].lower()
  
sw = set(stopwords.words('english')) 
#sw=set({'during', 're', 'mustn', 'wouldn', "wouldn't", 'now', "wasn't", 'how', 'all', 'our', 'herself', 'against', 'couldn', 'in', "mustn't", "you've", 'should', 'under', 'won', 'and', 'having', 'been', 'or', 'just', 'mightn', 'its', 'm', 'own', 'there', 'below', 'while', 'do', 'of', 'after', "didn't", 'him', 'those', 'whom', 'only', 'again', 'hasn', 'himself', 'are', 'wasn', 'that', 'doing', 'shan', 'it', 'y', 's', "don't", "weren't", 'hadn', 'my', 'yours', 'very', 'up', 'have', 'these', 'between', "you'd", 'until', 'for', 'them', 'then', 'myself', 'is', "shouldn't", "you're", "she's", 'before', 'such', 'no', 'off', 'she', 'we', 'ma', 'this', 'yourselves', 'his', 'an', 'ourselves', 'but', 'above', "it's", 'doesn', 'through', 'here', 'shouldn', 'when', 'out', 'does', 've', 'once', "isn't", 'am', 'being', 't', 'at', 'hers', 'same', "couldn't", 'few', "shan't", 'their', 'most', 'll', "won't", 'which', 'with', 'why', 'o', 'more', 'than', 'isn', 'did', 'further', 'other', 'didn', 'as', 'if', 'me', 'to', 'on', 'into', 'don', "doesn't", 'from', 'some', "you'll", 'd', 'itself', 'will', 'haven', 'be', "hasn't", 'about', 'your', "hadn't", 'aren', "that'll", 'has', 'any', 'weren', 'needn', "haven't", 'so', 'were', "aren't", 'what', 'a', 'by', 'not', 'had', 'nor', 'they', 'theirs', 'ain', 'i', 'can', 'you', "mightn't", 'over', 'because', 'he', 'themselves', 'was', 'each', 'too', "needn't", 'ours', 'where', 'down', 'her', 'the', 'both', "should've", 'yourself', 'who'})
krovetzstemmer=PyKrovetzStemmer()
#tokenizer=tokenizer()
N=3213835

k1=1.6
b=0.75
#t2=time.time()
#print(t2-t1)

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
    #print(tokens)
    queries[q]=tokens

#t3=time.time()
#print(t3-t2)

if model=='uni':
    Finalresult={}

    #print(time.time()-t3)
    #j=0
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
        
        
        '''
        for word,posting in index.items():
            
            
            for docid,tf in posting.items():
                dl=topdocs[q][docid]["length"]
                mu=3
                lamda=mu/(dl+mu)
                pc=sum(posting.values())/dlavg
                posting[docid]=round(math.log((tf+mu*pc)/(mu+dlavg),4))
                
        '''
        ##############################Query processing
        tokens=queries[q]
        
        score={}
        
        mu=dlavg*3
        c=len(index)
        # or c=sum([sum(posting.values()) for posting in index])
        for word in tokens:
            if(len(word)==1):
                continue
            if(word not in index):
                
                continue
           
            posting=index[word]
            pc=sum(posting.values())/c
            for docid in val:
                if(docid in posting):
                    dl=topdocs[q][docid]["length"]
                    if(dl==0):
                        continue
                    tf=posting[docid]
                    score[docid]=score.get(docid,0)+round(math.log((tf+mu*pc)/(mu+dl)),4)
                else:
                    dl=topdocs[q][docid]["length"]
                    if(dl==0):
                        continue
                    score[docid]=score.get(docid,0)+round(math.log(mu*pc/(mu+dl)),4)
        #Fullscore[q] = max(Fullscore[q].items(), key= lambda x: x[1])
        #Fullscore[q]=sorted(Fullscore[q].items(),key= lambda x: -x[1])
        
        if(len(score)==0):
            Finalresult[q]=(list(topdocs[q].keys())[0],0)
            continue
        
        Finalresult[q]=max(score.items(), key= lambda x: x[1])
        #print((time.time()-t),j)
        
    
    tem=sorted(list(Finalresult.keys()))    
    f = open("result_uni.txt", "w")
    for q in tem:
        
        a=Finalresult[q]

        f.write(str(q)+" Q0 "+a[0]+" "+str(1)+" "+str(a[1])+" t2est"+'\n' )
else:
    Finalresult={}
    
   
    for q,val in topdocs.items():
        ###########indexing
       
     
        index={}    
        dlavg=0
        for docid in val:
            tokens=tokenizer(doclist[docid])
            tokens=[word for word in tokens if word not in sw]
            tokens=list(map(krovetzstemmer.stem,tokens))
            topdocs[q][docid]["length"]=len(tokens)
         
            
            dlavg+=len(tokens)
            mem=''
            for word in tokens:
                if(len(word)==1):
                    continue
                if(len(mem)>0):
                    bigramword=mem+' '+word
                    
                    if bigramword in index:
                        index[bigramword][docid]=index[bigramword].get(docid,0)+1
                    else:
                        index[bigramword]={docid:1}
                    
                
                if word in index:
                    index[word][docid]=index[word].get(docid,0)+1
                    
                else:
                    index[word]={docid:1}
                mem=word 
        dlavg/=len(topdocs[q])
        c=len(index)
        '''
        for word,posting in index.items():
            
            
            for docid,tf in posting.items():
                dl=topdocs[q][docid]["length"]
                mu=3
                lamda=mu/(dl+mu)
                pc=sum(posting.values())/dlavg
                posting[docid]=round(math.log((tf+mu*pc)/(mu+dlavg),4))
                
        '''
        ##############################Query processing        
        tokens=queries[q]
        
        score={}
        u=3
        mu=u*dlavg
        mem=''
        for word in tokens:
            if(len(word)==1):
                mem=''
                continue
            if(word not in index):
                mem=''
                continue
            posting=index[word]
            c2=sum(posting.values())
            for docid in val:
                
                dl=topdocs[q][docid]["length"]
                f2=posting.get(docid,0)
                if(dl==0):
                    continue
                if(len(mem)==0):
                    score[docid]=score.get(docid,0)+round(math.log((f2+mu*c2/c)/(mu+dl)),4)
                else:
                    l=mu/(dl+mu)
                    m=1-l
                    bigram=mem+' '+word
                    if(bigram in index):
                        biposting=index[bigram]
                        c21=sum(biposting.values())
                        f21=biposting.get(docid,0)
                        
                        
                        
                    else:
                        f21=0
                        c21=0
                    f2=max(f2,1)
                    score[docid]=score.get(docid,0)+round(math.log(m*(m*f21/f2+l*f2/dl)+l*(m*c21/c2+l*c2/c)),4)
                    
                
            mem=word        
        #Fullscore[q] = max(Fullscore[q].items(), key= lambda x: x[1])
        #Fullscore[q]=sorted(Fullscore[q].items(),key= lambda x: -x[1])
        
        if(len(score)==0):
            Finalresult[q]=(list(topdocs[q].keys())[0],0)
            continue
        
        Finalresult[q]=max(score.items(), key= lambda x: x[1])
        
    tem=sorted(list(Finalresult.keys()))    
    f = open("result_bi.txt", "w")
    for q in tem:
        
        a=Finalresult[q]

        f.write(str(q)+" Q0 "+a[0]+" "+str(1)+" "+str(a[1])+" t2est"+'\n' )
    
