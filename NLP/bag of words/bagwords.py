import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re # For Cleaning the text
import nltk # library for Natural Language Processing


# BAG OF WORDS

#Importing the dataset

# helps remove everything from text except alphabet. the '^a-zA-Z' will be kept in text only, second param is the space before and after each word ,third param is the text
dataset = pd.read_csv('C:\\Users\\Rizwan\\Desktop\\MachineLearning\\Machine Learning A-Z Template Folder\\Part 7 - Natural Language Processing\\Section 36 - Natural Language Processing\\Restaurant_Reviews.tsv',
                      delimiter='\t' , quoting = 3) #quoting parameter is to ignore double quotes - 3 is the value to ignore double quotes in text

#using nltk library and we use it to remove unimportant words (preposition words like the,that,they,is etc)
#nltk.download('stopwords') # stopwords is a list provided by NLTK that has all unimportant words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


#Cleaning the text

corpus = []     # This is where we store the new clean text data

for i in range(0, len(dataset)):
        review = re.sub('[^a-zA-Z]' , ' ',dataset['Review'][i])
        review = review.lower()
        review = review.split()# converts string into list of words e.g "hi how are you" -> "hi","how","are","you"
        porter_stemmer = PorterStemmer()
        ## This for loops removes all words from list that are not needed
        #review = [porter_stemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]
        for i in list(review):
                if i in set(stopwords.words('english')):
                        review.remove(i)
        for i in range(0, len(review)):
                review[i] = porter_stemmer.stem( review[i])# Stemming process: word changes from loved to love
        review = ' '.join(review) # converted the list back to string
        corpus.append(review)

# Bag of words model


#
## Importing the dataset
#dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#
## Cleaning the texts
#import re
#import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#corpus = []
#for i in range(0, 1000):
#    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
#    review = review.lower()
#    review = review.split()
#    ps = PorterStemmer()
#    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#    review = ' '.join(review)
#    corpus.append(review)
#
## Creating the Bag of Words model
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features = 1500)
#X = cv.fit_transform(corpus).toarray()
#y = dataset.iloc[:, 1].values
#
## Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
#
## Fitting Naive Bayes to the Training set
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(X_train, y_train)
#
## Predicting the Test set results
#y_pred = classifier.predict(X_test)
#
## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)