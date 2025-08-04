                                                                                      Project Overview


Ner system is used to identify entities like person names, organization, and locations in news article. My project work on ner dataset having sentence tag Poc.
This project first clean the news article and then by using sklearn.curfuside. 
It work on real time news article 

2-Folder Structure

------NER using CRF.ipynb-----model file(Backend)
-----readme.txt(explanation)
-----front.py(Streamlit frontend)

3-Libraries and environment setup

1> Pandas
2>sklearn.model_selection
3>sklearn_crfsuite import CRF
4>sklearn_crfusite.metrics import flat_f1_score
5>sklearn_crfsuite.metrics import flat_classification_report


Fronend
1-streamlit
2-joblib
3-nltk
4-nltk.tokenize 

4-Can run backend directly using NER using CRF.ipynb file on jupyter notebook 

a>Input- can use whole but  it may take time so instead of using whole data using sample sata only by using df.sample it extract sample data itself from the whole dataset

b>Output-getting accuracy score and report 

6-can test with new data using this NER using CRF.ipynb as backend

can run on frontend also