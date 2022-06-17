# Assignment 2

In this assignment, we work with the **MS MARCO dataset** (https://microsoft.github.io/msmarco/), where unlike other datasets, the questions correspond to actual search queries that users submitted to Bing, and therefore may be more representative of a natural distribution of information need that users may want to satisfy using, say, an intelligent assistant. Our aim is to rerank the top-100 relevant documents for a given set of queries using **Probabilistic Retrieval Query Expansion**
and **Probabilistic Retrieval Query expansion**. More details on the problem statement in **Assignment2.pdf**. More details on the models implemented and results are given in **Algorithmic Details.pdf**.

### Libraries Required

Following libraries are required:

1)nltk
2)krovetz

### Commands to Run the Code

**1. python3 prob_rerank.py (query-file) (top-100-file) (collection-file) (expansion-limit)**  

On running the prob_rerank.py script, results for each term expansion will be stored in separate files in the trec_eval format.
Total m files are stored where m is the expansion limit. Files are named as result_pbx.txt  where x= # terms added (Note that x belongs to (1,2..m))


**2. python3 lm_rerank.py (query-file) (top-100-file) (collection-file) (model=uni|bi)**

On running the lm_rerank.py script:

For the unigram model, result file in the trec_eval format is stored as result_uni.txt. For the bigram model, result file in the trec_eval format is stored as result_bi.txt

**query-file**: file containing the queries in the same tsv format as given in the ms marco dataset

**top-100-file**: a file containing the top100 documents in the same format as train and dev top100 files given in the msmarco dataset, which need to be reranked

**collection-file**: file containing the full document collection (in the same format as msmarco docs file given)

**expansion-limit**: is a number ranging from 1—15 that specifies the limit on the number of additional terms in the expanded query

**model=uni|bi**: it specifies the unigram or the bigram language model that should be used for relevance language model.

