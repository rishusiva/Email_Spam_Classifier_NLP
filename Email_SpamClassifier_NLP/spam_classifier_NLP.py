import pandas as pd

# Gathering dataset and importing necessary modules

df = pd.read_csv('C:/Users/rishu/ML_Codebasics/Email_SpamClassifier/emails.csv')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Data Cleaning and Preprocessing

ps = PorterStemmer()
corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['text'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
#print("X=",X)

y = df.spam
#print("y=",y)

# Dividing the dataset into train and test data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#print(len(X_train))
#print(len(X_test))

# Naive Bayes classifier

from sklearn.naive_bayes import MultinomialNB
Model = MultinomialNB().fit(X_train, y_train)

y_pred = Model.predict(X_test) 

# Performance
print(Model.score(X_test,y_test))