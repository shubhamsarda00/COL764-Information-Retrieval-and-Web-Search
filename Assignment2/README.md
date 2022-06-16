# Assignment 2

For Running on hpc:
1. Activate the environment using command: 

 conda activate /home/textile/btech/tt1180958/myenv 

It already has all the required libraries installed:
a)nltk
b)krovetz
c)sys
d)math

Now simply run the commands specified in pdf for running the python scripts:

->  python3 prob_rerank.py [query-file] [top-100-file] [collection-file] [expansion-limit]

->  python3 lm_rerank.py [query-file] [top-100-file] [collection-file] [model=uni|bi]


The name of probabilistic reranking script is prob_rerank.py

The name of language model reranking script is lm_rerank.py

Structure of submission:

Entry_Num.zip->Entry_Num->[python scripts, readme and algorithmic description pdf]

where Entry_Num=2018TT10958


##########IMPORTANT################

After we run the prob_rerank.py script, results for each term expansion will be stored in separate files in the trec_eval format-

Total m files are stored where m is the expansion limit. Files are named as result_pbx.txt  where x= # terms added (Note that x belongs to [1,2..m])

For the unigram model, result file in the trec_eval format is stored as result_uni.txt

For the bigram model, result file in the trec_eval format is stored as result_bi.txt

NDCG, MRR and stat tests are reported in the pdf!
