# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 00:42:22 2020

@author: hp
"""
import nltk
import numpy as np
import os
import sys
#import time
import pickle,bz2

from nltk.tokenize import word_tokenize as tokenizer
#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
#from nltk.stem import PorterStemmer
#from nltk.tag.stanford import StanfordNERTagger
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('wordnet')

def is_num(a):
    a=a.replace(',','').replace('.','').replace('-','')
    
    try:
        num=float(a)
        return True
    except:
        return False
    
    
#try spacy and other tokenization methods
#file= open("TaggedTrainingAP/ap880212_t")
#t1=time.time()
#path = 'TaggedTrainingAP/'
path=sys.argv[1]
indexfile=sys.argv[2]
docs = os.listdir(path)
inverted={}
num_docs=0
file_num=-1
#stop_words = set(stopwords.words('english'))
#stop_words.add("''")
#stop_words.add('``')
#stop_words.add('""')
#stemmer=PorterStemmer()
#try using file.readlines() for faster
for doc in docs:
    file_num+=1
    #print(file_num)
    file = open(path+doc, 'r')
    while True:
        s=file.readline()
        if(len(s)==0):
            break
        docid=file.readline().split()[1]
        num_docs+=1
        while True:
            text=tokenizer(file.readline().lower())
            #text=nlp(file.readline().lower()).text.split()
            #text=file.readline()
            if(len(text)<=1):
                continue
            '''
            try:
                a=text[0],text[1]
            except:
                print(text)
            '''
            if(text[0]== "<" and text[1]=="TEXT"):
                
                text=tokenizer(file.readline().lower())
                #text=nlp(file.readline().lower()).text.split()
                ####stopword removal
                #text=[x for x in text if not x in stop_words]
                #print(text)
                ####
                n=len(text)
                i=0
                while i<n:
                    word=text[i]
                    if(word=="<"):
                        j=0
                        tag=''
                        memory=''
                        while True:
                            if(j>0):
                                i+=1 #for "<" 

                            i+=1
                            if(i>=n):
                                break
                            word=text[i]
                            #replace by p:,l:,o:
                            if word in ["person","organization","location"]:

                                #s="<"+word+text[i+1]+" "+text[i+2]+" "+text[i+3]+text[i+4]+text[i+5]
                                if(i+2>=n):
                                    break
                                s=word[0].upper()+":"+text[i+2]
                                #s=word[0].upper()+":"+stemmer.stem(text[i+2])
                                if j==0:

                                    #memory="<"+word+text[i+1]+" "+text[i+2]
                                    memory=word[0].upper()+":"+text[i+2]
                                    #memory=word[0].upper()+":"+stemmer.stem(text[i+2])
                                    tag=word
                                elif(j>0):
                                    memory+=" "+text[i+2]
                                    #memory+=" "+stemmer.stem(text[i+2])
                                if(s in inverted):
                                    inverted[s][docid]=inverted[s].get(docid,0)+1
                                    
                                else:
                                    inverted[s]={docid:1}
                                '''############adding word without tag
                                if(s[2:] in inverted):
                                    inverted[s[2:]][docid]=inverted[s[2:]].get(docid,0)+1
                                else:
                                    inverted[s[2:]]={docid:1}
                                ############    
                                '''    
                                i+=5
                                if(i+2>=n):
                                    break

                                if(i+2<n and text[i+2]==tag):

                                    j+=1
                                    continue
                                elif(j>0):
                                    #memory+=" </"+tag+">"
                                #add code for indexing new, delhi and not just L:new,L:Delhi,L:New Delhi
                                    
                                    if(memory in inverted):
                                        inverted[memory][docid]=inverted[memory].get(docid,0)+1
                                    else:
                                        inverted[memory]={docid:1}
                                    
                                    
                                    tag=''
                                    memory='' 
                                    j=0
                                    break
                                elif(j==0):
                                    tag=''
                                    memory=''
                                    break


                    elif(len(word)==1):
                        pass
                    else:
                        if(is_num(word)): #if(is_num(word) or word in stop_words)
                            pass
                        
                        else:
                            #########
                            #word=stemmer.stem(word)    
                            #########
                            if(word in inverted):
                                inverted[word][docid]=inverted[word].get(docid,0)+1
                            else:
                                inverted[word]={docid:1}

                            
                    i+=1



                next(file) #skipping </TEXT>
                '''
            elif(text[0]=="<" and text[1]=="HEAD" and text[-1]=='>'):
                i=3
                l=len(text)
                while(i<l and text[i]!='<'):
                    word=text[i].lower()
                    if(len(word)==1):
                        pass
                    else:
                        if(is_num(word)): #if(is_num(word) or word in stop_words)
                            pass
                        
                        else:
                            #########
                            #word=stemmer.stem(word)    
                            #########
                            if(word in inverted):
                                inverted[word][docid]=inverted[word].get(docid,0)+1
                            else:
                                inverted[word]={docid:1}
                    i+=1
                    '''        
            
            
            elif(text[0]=="<" and text[1]=="/DOC"):
                
                break
            
            else:
                continue

'''                
#####bigram addition
for name in docs:
    file_num+=1
    print(file_num)
    file = open(path+name, 'r')
    while True:
            text=tokenizer(file.readline())
            #text=file.readline()
            if(len(text)<=1):
                continue

            if(text[0]== "<" and text[1]=="TEXT"):
                text=tokenizer(file.readline().lower())
                ####stopword removal
                #text=[x for x in text if not x in stop_words]
                #print(text)
                ####
                n=len(text)
                i=0
                while i+1<n:
                    word=text[i]
                    
                    if(word=="<"):
                        i+=3
                        
'''                    
                

############
#d=dict(zip(sorted(inverted.keys()),range(len(inverted.keys()))))
#d=[(a,i) for (i,a) in list(enumerate(sorted(inverted.keys())))]
#d=sorted(inverted.keys())
file.close()
#t2=time.time()-t1
#print(t2)


#t3=time.time()
norms={}
for doc_list in inverted.values():
    n=len(doc_list)
    inv_doc_freq=round((np.log2(1+num_docs/n)),2)
    doc_list["inv_doc_freq"]=inv_doc_freq
    for key,val in doc_list.items():
        if(key=="inv_doc_freq"):
            continue
        w=round(round((1+np.log2(val)),4)*inv_doc_freq,2)
        doc_list[key]=w
        norms[key]=round(norms.get(key,0)+w**2,4)
for k,v in norms.items():
    norms[k]=round(np.sqrt(v),2)
#print(time.time()-t3)

#s = bz2.BZ2File("indexfile.dict", 'rb')
#d=pickle.load(s,encoding='latin')
#s.close()
#t=time.time()
s=bz2.BZ2File(indexfile+'.dict','w')
#pickle.dump(d,s,protocol=pickle.HIGHEST_PROTOCOL)
#pickle.dump(sorted(inverted.keys()),s,protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(sorted([(a,len(b)-1) for (a,b) in inverted.items()],key=lambda item:item[0]),s,protocol=pickle.HIGHEST_PROTOCOL)
s.close()
#a = bz2.BZ2File("indexfile.idx", 'rb')
#inverted=_pickle.load(a)
#norms=_pickle.load(a)
#inv_doc_freqs=_pickle.load(a)
a=bz2.BZ2File(indexfile+'.idx','w')    
pickle.dump(inverted,a,protocol=pickle.HIGHEST_PROTOCOL)
#pickle.dump(inv_doc_freqs,a,protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(norms,a,protocol=pickle.HIGHEST_PROTOCOL)
a.close()
#print(time.time()-t)
