# Importing the required libraries
import numpy as np
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
########################################################################

# Importing the dataset
dataset = pd.read_csv('dataset/Tweets.tsv', delimiter = '\t')


#########################################################################
# Function to remove the emojis from the input tweet
#Referred from https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b#gistcomment-3315605
def remove_emo(string):
    pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return pattern.sub(r'', string)


#########################################################################
# Function to print sentiments of the sentence by VADER
def vaderSentiment(sentence):

    # Creating  Sentiment Intensity Analyzer object.
    siaObject = SentimentIntensityAnalyzer()

    actualScore = siaObject.polarity_scores(sentence)
    # Polarity score provides a sentiment dictionary which contains pos, neg, neu and the compound scores

    # decide sentiment as positive, negative and neutral
    if actualScore['compound'] >= 0.05 :
        final=1

    elif actualScore['compound'] <= - 0.05 :
        final=0

    else :
        final=-1
    return final

#########################################################################

# Start reading the dataset and perform text cleaning
lemmatizer = WordNetLemmatizer()
corpus = []
for i in range(0,8426):
  tweet = re.sub('[^a-zA-Z]', ' ', dataset['Tweets'][i]) # Any element which is not a letter is replaced by space
  tweet = tweet.lower() # Convert to lower case
  tweet = tweet.split() #Splitting
  stopWords = stopwords.words('english') # Storing the English stopwords
  # Removing the words because it might lead to incorrect prdiction
  stopWords.remove('not') # it will not include the 'not' word in stopwords
  stopWords.remove('who')
  stopWords.remove('from')
  stopWords.remove('no')
  tweet = [lemmatizer.lemmatize(word) for word in tweet if not word in set(stopWords)] # Lemmatizing  only those words which will be helpful for the actual prediction
  tweet = ' '.join(tweet)
  remove_emo(tweet)  # Removing the emojis from the text input
  corpus.append(tweet)
#print(corpus)

#########################################################################


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)  #this is the size we want for our sparse matrix.max_features has an impact on the accuracy
X = cv.fit_transform(corpus).toarray() #fit_transform will fit corpus to X. Matrix of features must be 2d array
y = dataset.iloc[:, -1].values #Dependent variable column

#########################################################################

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#########################################################################

#Building the RandomForestClassifier Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 8, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train) #It will train the model on the training set

#########################################################################

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("Test set result")
length=len(y_pred)
length1=len(y_test)
firstArg=y_pred.reshape(length,1) #Vector of predicted profit. Reshape vector from horizontal to vertical
secondArg=y_test.reshape(length1,1)#Vector of Actual profit. Reshape vector from horizontal to vertical
print(np.concatenate((firstArg,secondArg),1))  #concatenate vertically the two vectors

#########################################################################

# Evaluating the performance
from sklearn.metrics import  classification_report,confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("")
ac=accuracy_score(y_test, y_pred)
print("Accuracy of the model is ",ac)
print("")
print(classification_report(y_test,y_pred))

#########################################################################

# Take input from user and predict the sentiment of the tweet
count=1
print("press 0 to exit and 1 to continue")
while count==1:      # loop to allow users to tweet multiple times
    tweet=input("Tweet your thoughts: ")
    vaderAnswer=vaderSentiment(tweet)   # Passing the same input to VADER Sentiment Analyzer
    tweet = re.sub('[^a-zA-Z]', ' ', tweet)
    tweet = tweet.lower()
    tweet = tweet.split()
    lemmatizer = WordNetLemmatizer()
    stopWords = stopwords.words('english')
    stopWords.remove('not')
    stopWords.remove('who')
    stopWords.remove('from')
    stopWords.remove('no')

    tweet = [lemmatizer.lemmatize(word) for word in tweet if not word in set(stopWords)]
    tweet = ' '.join(tweet)
    remove_emo(tweet)
    new_corpus = [tweet]
    new_X_test = cv.transform(new_corpus).toarray()
    new_y_pred = classifier.predict(new_X_test)
    new=new_y_pred[0] # Getting clean representation of the output
    print(" Prediction of ML model : ",new)
    if new==1:
        print("Positive")
    elif new==0:
        print("Negative")
    print("")
    if vaderAnswer==1:
        print("Vader:Positive")
    elif vaderAnswer==0:
        print("Vader:Negative")
    else:
        print("Vader:Neutral")

    to_end=input(" Do you want to continue ")
    to_end=int(to_end)
    if to_end==0:
        count=0
    else:
        count=1

######################## End of Code #################################################
