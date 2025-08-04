                             Named Entity Recognition (NER) for News Articles

1. Project Overview

This project performs Named Entity Recognition on news articles to extract named entities like persons, organizations, and locations using a BiLSTM deep learning model.
Named Entity Recognition (NER) is a subtask of Natural Language Processing (NLP) that identifies and classifies named entities in text into predefined categories.
How it operates:
 Large text datasets with named entities already classified into their appropriate categories serve as the training data for NER models. 
 Machine Learning: The NER models are trained using a variety of machine learning approaches, such as rule-based approaches, statistical models, and deep learning. 
 Contextual Understanding: Because certain words can have more than one meaning, NER models examine the context of words and phrases to distinguish entities and choose the best category.


2. Folder Structure in the project

- ner.csv: It is the dataset file where data is stored
- train.ipynb: Trains and saves BiLSTM model in the train_model file
- app.py : It is for User Interface which is shown on streamlit
- requirements.txt: All required Python packages

3. Libraries & Environment Setup

pip install -r requirements.txt
pip install pandas
pip install numpy
pip install sklearn-crfsuite


4. How to Run the Code

- To run the code write commands in the command prompt:
  Open jupyter notebook then open train.ipynb file then run all the code from the kernels

- To run the User Interfacewrite commands in the command prompt:
  streamlit run app.py

5. Input & Output Explanation

The input of the code expects in the text.
