# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 00:53:28 2020

@author: hp
"""
import sys
import pickle,bz2


s = bz2.BZ2File(sys.argv[1], 'rb')
d=pickle.load(s)
s.close()
i=0
for w in d:
    print(w[0]+":"+str(w[1])+":"+str(i))
    i+=1