1. Project Overview:
 
    The project is Sentiment Analysis of Product Reviews. 
    In this project a model is been train with some data and and tested over it.
    When a user will provide a reviwe for a product as a text, model will predict the sentiment of the 
    review as it is a Postive review, Peutral review or Negative review 

2. Folder Stucture:

    a. I have create a root folder named SumitSingh_Question2.
    b. In this folder I have create jypyter file named as Sentiment_Analysis.ipynb, in which i have written the code for the model
        and saved model as pickle file.
    c. The model saved file name is sentiment_model.pkl
    d. After that i have created a sentiment_analysis.py file in which I have written the code for the user interface and
        I have import the saved model in this file.

3. Libraries and Environment Setup:

    Python Libraries:
        Pandas, sklearn, nltk, re, warnings
    Tools:
        stopwords, sent_tokenize, word_tokenize, WordNetLemmitizer, TfidfVectorizer
    
    To install these libraries, run command pip install -r requirement.text

4. How to run code:
 
    a. I have written the main code file as .ipynb extension, so to run this file, run every cell after other.
    b. Next to run the user interface of the model, run command 'streamlit run sentiment_analysis.py'.
    c. Opent http://localhost:8501/ on browser for the user interface.


5. Input and output Explanation:

    a. In user interace we will provdie a review according to the user review. This input will be in text format
        and it could involve number and special character.
    b. For the output there is a button on user interface 'Predict Sentiment'
       After clicking on this button the output will appear.
    c. Output will be either 'Postive Review', 'Neutral Review', or 'Negative Review'.

6. How to Test with New data

    To test the model over the new data or unseen data-
    a. If we are test model over new data in .ipynb file, then follow the steps-
       -> print(model.predict("Testing Text"))
    
    b. If we are testing the model over new data through user interface, then follow the steps-
        -> In user interface there is text box, in which we have to write the text to which we have to
          test the model
        -> After typing the text or review, click on predict sentiment button.
           Prediction for the new test data will apper below
