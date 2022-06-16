# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 01:04:57 2020

@author: hp
"""

########IMP
import os,sys
java_path=""
try:
    java_path = sys.argv[12]
except:
    java_path = "C:/Program Files/Java/jdk-11.0.3/bin/java.exe"
        
os.environ['JAVAHOME'] = java_path
os.environ['STANFORD_MODELS'] = 'stanford-ner-4.0.0'

###########

import nltk
import numpy as np
import random
import pickle,bz2

from nltk.tokenize import word_tokenize as tokenizer
#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
#from nltk.stem import PorterStemmer
from nltk.tag.stanford import StanfordNERTagger
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


######setting environmwnt



#t5=time.time()
def is_num2(a):
    a=a.replace(',','').replace('.','').replace('-','')
    try:
        num=float(a)
        return True
    except:
        return False
    
def binary_search(a, b): 
    l=m=0
    h = len(a) - 1
  
    while l <= h: 
  
        m = (l+h)//2
        if a[m][0].startswith(b):
            return m
        elif b>a[m][0]: 
            l=m+1
  
        else:
            h=m-1
    return -1    
#try using file.readlines() for faster
s = bz2.BZ2File(sys.argv[10], 'rb')
d=pickle.load(s)
s.close()
a = bz2.BZ2File(sys.argv[8], 'rb')
inverted=pickle.load(a)
#inv_doc_freqs=pickle.load(a)
norms=pickle.load(a)
#print(d[79][0])

a.close()


queries=open(sys.argv[2])
cut_off=int(sys.argv[4])

ans=[]
os.environ['STANFORD_MODELS'] = 'stanford-ner-4.0.0'
stanford_tagger = StanfordNERTagger('stanford-ner-4.0.0/classifiers/english.all.3class.distsim.crf.ser.gz',
               'stanford-ner-4.0.0/stanford-ner-4.0.0.jar')
num_terms=len(d)
qids=[]
tag=[0,0]
while True:
    text=queries.readline()
    
    if(len(text)==0):
        break

    if("<num>"==text[0:5]):
        
        qids.append(text.split()[2])
        #print(qids)
        continue
        
    if "<title>"==text[0:7]:
        text=text.replace('/',' ')
        #text=text.replace('-',' ')
        text=tokenizer(text)    #maybe try skipping this step and do st.tag(text.split())
        
        text=stanford_tagger.tag(text[5:])  #skip first 5 as ['<','title','<','topics',':']
        #print(text)
        n=len(text)
        q={}
        for i in range(n):
            term=text[i][0]
            if(tag[0]>0):
                tag[0]-=1
                if(tag[0]==0):
                    if(term[-1]!='*'):
                        if(tag[1]=='N'):
                            q['P:'+term.lower()]=q.get('P:'+term.lower(),0)+1
                            q['L:'+term.lower()]=q.get('P:'+term.lower(),0)+1
                            q['O:'+term.lower()]=q.get('P:'+term.lower(),0)+1

                        else:
                            q[tag[1]+":"+term.lower()]=q.get(tag[1]+":"+term.lower(),0)+1
                    else:
                        term=term[:-1]
                        if(tag[1]=='N'):
                            idx=binary_search(d,term)
                            if(idx==-1):
                                pass
                            else:
                                g=idx
                                while(g>=0 and g<num_terms and d[g][0].startswith(term)):
                                    q["P:"+d[g][0]]=q.get("P:"+d[g][0],0)+1
                                    q["O:"+d[g][0]]=q.get("O:"+d[g][0],0)+1
                                    q["L:"+d[g][0]]=q.get("L:"+d[g][0],0)+1
                                    g-=1
                                g=idx+1
                                while(g>=0 and g<num_terms and d[g][0].startswith(term)):
                                    q["P:"+d[g][0]]=q.get("P:"+d[g][0],0)+1
                                    q["O:"+d[g][0]]=q.get("O:"+d[g][0],0)+1
                                    q["L:"+d[g][0]]=q.get("L:"+d[g][0],0)+1
                                    g+=1
                            
                        else:
                            idx=binary_search(d,term)
                            if(idx==-1):
                                pass
                            else:
                                g=idx
                                while(g>=0 and g<num_terms and d[g][0].startswith(term)):
                                    q[tag[1]+":"+d[g][0]]=q.get(tag[1]+":"+d[g][0],0)+1
                                    g-=1
                                g=idx+1
                                while(g>=0 and g<num_terms and d[g][0].startswith(term)):
                                    q[tag[1]+":"+d[g][0]]=q.get(tag[1]+":"+d[g][0],0)+1
                                    g+=1
                            
                            
                        
                continue
                
            
            term=text[i]
            #if(term[0] in stop_words):
             #   continue
            terms=term[0].split('-')
            
            if(len(terms)==1):
                if(is_num2(term[0])):
                    continue
                elif(len(term[0])==1):
                    if term[0] in ['O','L','P','N']:
                        #print(1)
                        if(i+1<n and text[i+1][0]==':'):
                            tag[0]=2
                            tag[1]=term[0]
                            continue
                    else:
                        continue
                    
                    
                else:

                    t=False
                    word=''
                    if(term[1] in ['PERSON','LOCATION','ORGANIZATION']):
                        word=term[1][0]+':'+term[0]  #taking first letter of tag
                        #word=term[1][0]+':'+stemmer.stem(term[0])
                        t=True
                    else:
                        word=term[0]
                        #word=stemmer.stem(term[0])
                    if(word[-1]=='*'):
                        #prefix search  (for now continuing with hashmap henceiterating over whole vocab, otherwise we can also keep it in sorted order)
                        word=word[:-1].lower()   #remove last letter of string
                        idx=binary_search(d,word)
                        if(idx==-1):
                            pass
                        else:
                            g=idx
                            while(g>=0 and g<num_terms and d[g][0].startswith(word)):
                                q[d[g][0]]=q.get(d[g][0],0)+1
                                g-=1
                            g=idx+1
                            while(g>=0 and g<num_terms and d[g][0].startswith(word)):
                                q[d[g][0]]=q.get(d[g][0],0)+1
                                g+=1

                    else:

                        #q[word]=q.get(word,0)+1
                        if(t):
                            q[word[0]+word[1:].lower()]=q.get(word[0]+word[1:].lower(),0)+1
                            q[word[2:].lower()]=q.get(word[2:].lower(),0)+1   #may or may nor use
                        else:
                            q[word.lower()]=q.get(word.lower(),0)+1
            else:
                s=''
                for term in terms:
                    if(len(term)<2):
                        continue
                    else:
                        #word=stemmer.stem(term)
                        q[term.lower()]=q.get(term.lower(),0)+1
                        s+=term.lower()+"-"
                if(len(s)>1):
                    q[s[:-1]]=q.get(s[:-1],0)+1
        
        for k,v in q.items():
            try:
                #inv_doc_freq=inv_doc_freqs[k]
                inv_doc_freq=inverted[k]["inv_doc_freq"]
            except:
                inv_doc_freq=1
            
            q[k]=round(round(np.log2(1+v),4)*inv_doc_freq,4)
        #print(q)
        similarity={}
        norm_q=0
        for values in q.values():
            norm_q+=round(values**2,4)
        norm_q=round(np.sqrt(norm_q),4)    
        for term,w in q.items():
            if term in inverted:
                for doc,v in inverted[term].items():
                    if(doc[0]=='i'):#
                        continue###for 2nd type
                    similarity[doc]=round(similarity.get(doc,0)+v*w,4)
            else:
                pass
        for doc,v in similarity.items():
            similarity[doc]=round(v/(norms[doc]*norm_q),4)
        if(len(similarity)<cut_off):
            #simply sort
            similarity=sorted(similarity.items(),key= lambda x: -x[1])
            a=len(similarity)
    
            while a!=cut_off:
                a+=1
                similarity.append((list(norms.keys())[random.randint(0,len(norms))],0))
        else:
            #argpartition for top k (for now simple sorting)
            similarity=sorted(similarity.items(),key= lambda x: -x[1])[:cut_off]
        
        ans.append(similarity)
    else:
        continue
#t6=time.time()-t5 
f = open(sys.argv[6], "w")
qid=0
for query in ans:
    i=1
    for score in query:
        f.write(str(int(qids[qid]))+" Q0 "+score[0]+" "+str(i)+" "+str(score[1])+" t2est"+'\n' )
        
        i+=1
    qid+=1
f.close()        
    
    
    
    
