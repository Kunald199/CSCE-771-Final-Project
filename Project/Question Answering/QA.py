#### Importing the libraries #####
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
#########################################################################

# Importing the dataset
dataset = pd.read_csv('dataset/Cases.tsv', delimiter = '\t')  # Number of cases dataset
datasetMask = pd.read_csv('dataset/Mask.tsv', delimiter = '\t') # Mask support dataset
datasetDeaths = pd.read_csv('dataset/Deaths.tsv', delimiter = '\t') # Number of deaths dataset


#########################################################################

#### Code for mask support ######


corpusMask = []
for i in range(0, 922):
  rMask = re.sub('[^a-zA-Z]', ' ', datasetMask['County'][i]) # Any element which is not a letter is replaced by space
  rMask = rMask.lower() # Convert to lower case
  rMask = rMask.split() # Splitting
  ps = PorterStemmer() #Stemmer object
  stopwordsMask = stopwords.words('english') # Storing the English stopwords
  stopwordsMask.remove('not') # it will not include the 'not' word in stopwords
  rMask = [ps.stem(wordMask) for wordMask in rMask if not wordMask in set(stopwordsMask)] # Stem  only those words which will be helpful for the actual prediction
  rMask = ' '.join(rMask)
  corpusMask.append(rMask)
# print(corpusMask)

#########################################################################

# Creating the Bag of Words model

cvMask = CountVectorizer(max_features = 111) #this is the size we want for our sparse matrix.max_features has an impact on the accuracy
XMask = cvMask.fit_transform(corpusMask).toarray() #fit_transform will fit corpus to X. Matrix of features must be 2d array
yMask = datasetMask.iloc[:, -1].values # Result column
# print(yMask)

#########################################################################

# Splitting the dataset into the Training set and Test set

X_trainMask, X_testMask, y_trainMask, y_testMask = train_test_split(XMask, yMask, test_size = 0.25, random_state = 0)

#########################################################################

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifierMask = GaussianNB()
classifierMask.fit(X_trainMask, y_trainMask)

#########################################################################
# Predicting the Test set results
y_predMask = classifierMask.predict(X_testMask)
# print('Test set result is')
# print(np.concatenate((y_predMask.reshape(len(y_predMask),1), y_testMask.reshape(len(y_testMask),1)),1))
# For concatenate we first reshape the vectors vertically and concatenate
#########################################################################

# Making the Confusion Matrix

cmMask = confusion_matrix(y_testMask, y_predMask)
# print(cmMask)
acMask=accuracy_score(y_testMask, y_predMask)
# print(acMask)

#########################################################################
###### Code for answering questions about death count in county####

# Cleaning the texts

corpusDeaths = []
for iDeaths in range(0, 336):
  rDeaths = re.sub('[^a-zA-Z]', ' ', datasetDeaths['County'][iDeaths]) # Any element which is not a letter is replaced by space
  rDeaths = rDeaths.lower() # Convert the string to lower case
  rDeaths = rDeaths.split() #Splitting the string
  lemmatizer = WordNetLemmatizer() #lemmatizer object
  stopwordsDeaths = stopwords.words('english') # Storing the English stopwords
  stopwordsDeaths.remove('not') # it will not include the 'not' word in stopwords
  rDeaths = [lemmatizer.lemmatize(word) for word in rDeaths if not word in set(stopwordsDeaths)] #lematizing only those words who are not in the stopwords list
  rDeaths = ' '.join(rDeaths)
  corpusDeaths.append(rDeaths)
 # print(corpusDeaths)

#########################################################################

# Bag of words model

cvDeaths = CountVectorizer(max_features = 120)
XDeaths = cvDeaths.fit_transform(corpusDeaths).toarray()
yDeaths = datasetDeaths.iloc[:, -1].values

#########################################################################

# Splitting the dataset into the Training set and Test set

X_trainDeaths, X_testDeaths, y_trainDeaths, y_testDeaths = train_test_split(XDeaths, yDeaths, test_size = 1, random_state = 0)


#########################################################################

#Building the RandomForestClassifier Model
from sklearn.ensemble import RandomForestClassifier
classifierDeaths = RandomForestClassifier(n_estimators = 15, criterion = 'entropy', random_state = 0)
classifierDeaths.fit(X_trainDeaths, y_trainDeaths)

#########################################################################

# Predicting the Test set results
y_predDeaths = classifierDeaths.predict(X_testDeaths)
# print('Test set result is')
# print(np.concatenate((y_predDeaths.reshape(len(y_predDeaths),1), y_testDeaths.reshape(len(y_testDeaths),1)),1))

#########################################################################

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cmDeaths = confusion_matrix(y_testDeaths, y_predDeaths)
# print(cmDeaths)
acDeaths=accuracy_score(y_testDeaths, y_predDeaths)
# print(acDeaths)

################################  Deaths Code Done  ##########################################




###########      Main Menu   #############
count=1
while count==1:    # a loop so that user can ask multiple times
    print("Menu")
    print(" 1: Cases in county  2: Number of deaths 3: Mask support by population")
    choice=input()
    try:
        choice=int(choice)
    except:
        print("********Exception********")

    if choice==1:   # Tried to put this code above as the other code, but it was having an issue so decided to put it inside the loop
        # To find about cases in county

        # Cleaning the texts
        corpusCases = []
        for iCases in range(0, 639):
          rCases = re.sub('[^a-zA-Z]', ' ', dataset['County'][iCases])
          rCases = rCases.lower()
          rCases = rCases.split()
          lemmatizer = WordNetLemmatizer()
          stopwordsCases = stopwords.words('english')
          stopwordsCases.remove('not')
          rCases = [lemmatizer.lemmatize(word) for word in rCases if not word in set(stopwordsCases)]
          rCases = ' '.join(rCases)
          corpusCases.append(rCases)
        #print(corpusCases)

        # Creating the Bag of Words model

        cv = CountVectorizer(max_features = 120)
        X = cv.fit_transform(corpusCases).toarray()
        y = dataset.iloc[:, -1].values
        #print(y)

        # Splitting the dataset into the Training set and Test set

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        #print('Test set result is')
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

        # Making the Confusion Matrix

        cm = confusion_matrix(y_test, y_pred)
        #print(cm)
        ac=accuracy_score(y_test, y_pred)
        #print(ac)


        countCase=1

        while countCase==1:
            newCase=input("Enter question: ")
            newCase = re.sub('[^a-zA-Z]', ' ', newCase)
            newCase = newCase.lower()
            newCase = newCase.split()
            lemmatizer = WordNetLemmatizer()
            stopWordCase = stopwords.words('english')
            stopWordCase.remove('not')
            newCase = [lemmatizer.lemmatize(word) for word in newCase if not word in set(stopWordCase)]
            newCase = ' '.join(newCase)
            corpusCase = [newCase]
            XtestCase = cv.transform(corpusCase).toarray()
            yPredCase = classifier.predict(XtestCase)
            new=yPredCase[0]
            print("Your Answer is: ",new)
            print("The accuracy of the ML model is ",ac)

            print("Would you like to continue 1-yes 0-no ")
            choiceCases=input("Enter your choice ")
            try:
                choiceCases=int(choiceCases)
                if choiceCases==1:
                        countCase==1
                elif choiceCases==0:
                        countCase==0
                        break
            except:
                print("Please enter valid option")
                break

    if choice==2: # To find about deaths in county
        countCase=1
        while countCase==1:
            newDeath=input("Enter question ")
            print("Your Question is ",newDeath)
            newDeath = re.sub('[^a-zA-Z]', ' ', newDeath)
            newDeath = newDeath.lower()
            newDeath = newDeath.split()
            lemmatizerDeath = WordNetLemmatizer()
            stopDeaths = stopwords.words('english')
            stopDeaths.remove('not')
            newDeath = [lemmatizerDeath.lemmatize(word) for word in newDeath if not word in set(stopDeaths)]
            newDeath = ' '.join(newDeath)
            newCorpusDeaths = [newDeath]
            newXDeaths = cvDeaths.transform(newCorpusDeaths).toarray()
            newyDeaths = classifierDeaths.predict(newXDeaths)
            print(" Your answer is : ",newyDeaths[0])
            print("Accuracy of model is ",acDeaths)
            print("Would you like to continue 1-yes 0-no ")
            choiceCases=input("Enter your choice ")
            try:

                choiceCases=int(choiceCases)
                if choiceCases==1:
                    countCase==1
                elif choiceCases==0:
                    countCase==0
                    break
            except:
                print("Please enter valid option")
                break

    if choice==3: # To find about mask support in county
        countCase=1
        while countCase==1:
            newMask=input("Enter question ")
            print("Your Question is ",newMask)
            newMask = re.sub('[^a-zA-Z]', ' ', newMask)
            newMask = newMask.lower()
            newMask = newMask.split()
            ps = PorterStemmer()
            stopWordMasks = stopwords.words('english')
            stopWordMasks.remove('not')
            newMask = [ps.stem(word) for word in newMask if not word in set(stopWordMasks)]
            newMask = ' '.join(newMask)
            corpusMask = [newMask]
            XtestMask = cvMask.transform(corpusMask).toarray()
            yPredMask = classifierMask.predict(XtestMask)
            new1=yPredMask[0]/1000
            new=new1*100
            print(" Your answer is : ",new ," %")
            print(" Accuracy score of model is ",acMask)
            print("Would you like to continue 1-yes 0-no ")
            choiceCases=input("Enter your choice ")
            try:
                choiceCases=int(choiceCases)
                if choiceCases==1:
                    countCase==1
                elif choiceCases==0:
                    countCase==0
                    break
            except:
                print("Please enter valid option")
                break

    else:
        print(" ")



    mainChoice=input("Do you want to ask any other information 1=yes 0=no")
    try:
            mainChoice=int(mainChoice)
            if mainChoice==1:
                count==1
            elif mainChoice==0:
                count==0
                break

    except:
        print("******* Please enter valid integer **********")

################################### End of Code ######################################
