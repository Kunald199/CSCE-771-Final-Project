# CSCE-771-Final-Project

This project has two parts, 1: Sentiment Analysis 2: Question Answering

Approach for both the problem is same i.e. using ML model approach.

In each, Sentiment Analysis and Question Answering folder we have a dataset folder. It contains the dataset which will be used for the project.

Dataset of Sentiment Analysis : http://www.kgswc.org/hackathon-2020/
Dataset of Question Answering : https://github.com/nytimes/covid-19-data/blob/master/mask-use/mask-use-by-county.csv , https://github.com/nytimes/covid-19-data/blob/master/live/us-counties.csv


Manipulations have been done in dataset to fit them into the project scope.

For the mask support dataset, originally it contained percentages, but in the dataset we reversed it and stored as an integer number and inside the code we have calculated the percentage for simplicity.
*******************************************************************************
For this project we have used open source libraries and packages

sklearn
numpy 
re
nltk
vaderSentiment
pandas  

There is a seperate file which cites there references.

*******************************************************************************
For executing the file, download a zip in your local computer and execute the program in cmd

For Sentiment Analysis, write "Sentiment.py" in cmd to run the program and for Question Answering write "QA.py" in cmd  to run the program
Note: Ensure that you are in proper folder
