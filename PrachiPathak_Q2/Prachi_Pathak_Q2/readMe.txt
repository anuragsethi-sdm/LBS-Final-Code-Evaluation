SENTIMENT ANALYSIS OF PRODUCT REVIEWS

1. Project Overview:

Building a model to classify product reviews as Positive, negative or neural.
Here in this, build a model which predict the review as positive, negative and neural on the basis of the following given reviews by the user.

2. Folder structure:

-> Sentement_analysis_Product_Review.ipynb - is the file in which the model is build.
-> Product_review_analysis.py - is the file in which User Interface is build
-> amazon_reviews.csv - is the csv file or the data which is trained on the model

3. Libraries Used:

There are following Libraries are used in it which are as follows:

-> Pandas: pandas is used for the dataset.
-> nltk: Natural Language Toolkit used to work with human         language dataset.
-> re: Regular expression used for pattern.
-> stopword: used for the removing of breaking words.
-> word_tokenize: used to convert the data into tokens.
-> WordNetLemmatizer: use to convert the text into base form.
-> TfidfVectorizer: used to convert the text into numeric form for machine understandable.
-> Logistic Regression: use a Logistic Regression model for the classification of the text.
-> Accuracy_score: used for finding the accuracy score.
-> precision: used for finding the precision score.
-> recall: used for finding the recall.
-> f1-score: used for the finding of f1-score.
-> pickle: used for the saving the model into pickle.

4. To run the code:
On command prompt- run "streamlit run Product_review_analysis.py" 

5. Input and Output explaination:
-> for the input - csv file is uploaded from which the required columns are extracted for training the model.
-> for the output - it predict the result as positive, negative and neural , if the value is greater than 3 then it predict positive, and if the value is less than 3 then it predict negative, and if the value is 3 then it predict neutral.





