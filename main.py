#!/usr/bin/env python
# coding: utf-8

# In[1]:

from PyPDF2 import PdfFileReader, PdfFileWriter
import fitz
import numpy as np
import pandas as pd
import nltk
from sortedcontainers import *
from nltk import bigrams
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from pprint import pprint 
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer as wnl
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk import bigrams
import re
import math
import statistics
import time
from time import sleep 
import string
from nltk.tokenize import RegexpTokenizer

# In[2]:


fileName = input("Enter name of the file") 
doc = fitz.open(fileName+".pdf")
pageNo=1
pageData={}
for page in doc:
	text = page.getText()
	pageData[pageNo] = text
	pageNo = pageNo+1

docsData = {}
#All the necessary preprocessing steps needs to be done here
for key in pageData:
	docsData[key] = sent_tokenize(pageData[key])       
docID_pageID = {}    # Key: docID value: pageID
pages = []           # List of docID with index = docID    
docID = 0
sent = ""
for key in docsData:
	for sentence in docsData[key]:
		pages.append(sentence)
		docID_pageID[docID] = key
		docID = docID+1

#Final : pages : list of docs with index = docId
# docID_pageID : dictionary of key: docID value: pageNo


# In[3]:


def constructDoc(pages):
    tokenizer = RegexpTokenizer(r'\w+')
    page = []
    lst = []
    for sent in pages:
        words = tokenizer.tokenize(sent)
        new_words = []
        for word in words:
            if(word not in stop_words):
                new_words.append(word.lower())
        lst.append(new_words)
    return lst


# In[4]:


def calcIDF(docList, idf): 
    docFreq = SortedDict()   
#First Calculate docFreq for all the terms
    for doc in docList:
        done = SortedSet()
        for term in doc:
            if term not in docFreq:
                docFreq[term]=0
            if term not in done:
                docFreq[term]= docFreq[term]+1
                done.add(term)
    N = len(docList)
#Then idf is calculated
    for term, freq in docFreq.items():
        idf[term] = math.log10(N/freq)
    print(idf)
    return


# In[5]:


def getWeight(doc,idf, isQ):
    freq = SortedDict()
    tf = SortedDict()
    # print(doc)
    for term in doc:
        if term not in freq:
            freq[term]=0
        freq[term]+=1
    for term, f in freq.items():
        if term not in tf:
            tf[term]=0
        tf[term] = (1+math.log10(f))
    w = SortedDict()
    sparse = []
    mag = 0
    for term in doc:
        val = 0
        if term in idf:
            val = tf[term]*idf[term]
        if isQ:
            val=tf[term]
        w[term]= val
        sparse.append(val)
        mag = mag+ val*val
    

    length = math.sqrt(mag)
    #w is map<str,int>
    return w, length


# In[6]:


def getW_td(docList,idf, optVec, isQ):
# for every doc there will be a dictionary for term->weight
    w = [None]*(len(docList))
    vecs = []
#here w is vec<int,map<str,int>>
#vecs is list of document vectors
    docId=0
    for doc in docList:
        # print(docId ,end=" ")
        vecs.append([])
        w[docId],length = getWeight(doc,idf, isQ)
        if optVec:
            for term, val in idf.items():
                if term in w[docId]:
                    vecs[docId].append(w[docId][term]/length)
                else:
                    vecs[docId].append(0)
        docId+=1
    if optVec:
        vecs = list(map(np.asarray,vecs))
        return w,vecs
    else:
        return w


# In[7]:


def getScoreTable(queryList,score):
    scoreTable = pd.DataFrame(columns=["QueryId", "DocIDs", "Scores"])
    retrieved = SortedDict()
    for qid in range(len(queryList)):
        docsFound = []
        temp  = {}
        ids = ""
        sc = ""
        for a, b in score[qid]:
            if a==0:
                continue
            sc = sc + f' {a},'
            ids = ids + f' {b},'
            docsFound.append(b)
        temp["QueryId"] = qid+1
        temp["DocIDs"] = ids
        temp["Scores"] = sc
        scoreTable = scoreTable.append(temp, ignore_index = True)
        retrieved[qid]= docsFound
    return retrieved,scoreTable


# In[8]:


def calcVecScore(vecsDoc,q):
    ans=[]
    docId=0
    for d in vecsDoc:
        ans.append((np.dot(d,q), docId+1))
        docId+=1
    return sorted(ans,reverse=True)


# In[9]:


def getVecScore(queryList,vecsDoc, vecsQ):
#score should be map<qid, vec<pair<score, docId>>>
# Calcuation of Score for queries for a doc
    score= SortedDict()
    qid=0
    for q in queryList:
        score[qid] = []
        score[qid] = calcVecScore(vecsDoc, vecsQ[qid])
        score[qid] = score[qid][0:20] #tuple of <score,docid>
        qid+=1
    return score


# In[10]:


stop_words = SortedSet(stopwords.words('english'))
class main():
    def __init__(self):
        self.optVec=1
        self.docList = constructDoc(pages)
        self.query = input("Enter the query")
        self.doc_query = [self.query]
        self.queryList = constructDoc(self.doc_query)
        self.idf = SortedDict()
        calcIDF(self.docList, self.idf)
        self.wDoc,self.vecsDoc=getW_td(self.docList,self.idf, self.optVec, isQ=0)
        self.wQ,self.vecsQ = getW_td(self.queryList,self.idf, self.optVec, isQ=1)
        self.score = getVecScore(self.queryList,self.vecsDoc,self.vecsQ)
        self.retrieved,self.scoreTable = getScoreTable(self.queryList,self.score)


# In[11]:


task = main()


# In[12]:

resultDocIDs = task.scoreTable["DocIDs"].tolist()
res=""
for s in resultDocIDs:
    res = s.split(',')

result=[]
for x in res:
    if x!='':
        result.append(int(x)-1)
result = result[:10]
resultDocIDs = result
resultPagesNo = []
for docID in resultDocIDs:
    print(docID)
    resultPagesNo.append(docID_pageID[int(docID)])
    print(docID_pageID[int(docID)])
print(resultPagesNo)
resultPageData = []
for docID in resultDocIDs:
    resultPageData.append(pages[docID])
    print("* " + pages[docID])


#resultDocIDs. - doc Id of the query finding
#resultPagesNo - pageNumbers of the query finding
#resultPageData- doc Data of the query finding

# In[13]:


def purifyText(text):
    lst = text.split('-\n')
    result = ""
    for x in lst:
        result=result+x
    return result


# In[14]:


#print(resultPagesNo)


# In[15]:
#resultDocIDs. - doc Id of the query finding
#resultPagesNo - pageNumbers of the query finding
#resultPageData- doc Data of the query finding
for i in range(len(resultPagesNo)):
    page = doc[resultPagesNo[i]-1]

    ### SEARCH

    text = resultPageData[i]
    text=purifyText(text)
    print(text)
    text_instances = page.searchFor(text)
    print(text_instances)
    ### HIGHLIGHT

    for inst in text_instances:
        highlight = page.addHighlightAnnot(inst)


### OUTPUT

doc.save("Full Doc highlighted.pdf", garbage=4, deflate=True, clean=True)
# In[16]:


from PyPDF2 import PdfFileWriter,PdfFileReader

pdf1=PdfFileReader(open("./Full Doc highlighted.pdf", 'rb'))
writer = PdfFileWriter()
finResultPages = []
for i in range(len(resultPagesNo)-1):
    if resultPagesNo[i] not in finResultPages:
        finResultPages.append(resultPagesNo[i])
resultPagesNo = finResultPages
print(resultPagesNo)
for i in range(len(resultPagesNo)):
    writer.addPage(pdf1.getPage(resultPagesNo[i]-1))
    
# write to file
with open("destination.pdf", "wb") as outfp:
    writer.write(outfp)


# In[16]:
print(task.scoreTable.to_string())
print(task.docList[8])




