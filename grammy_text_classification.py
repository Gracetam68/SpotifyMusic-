#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 22:07:27 2021

@author: lichangtan
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read the file 
grammy = pd.read_csv('updated_attributes_csv.csv')
print(grammy.info())

# only select Name,Explicit,won grammy from the grammy dataset
grammy_1 = grammy[['Name','won grammy']]
grammy_1.isnull().sum()
grammy_1.shape # (154931, 3)
grammy_1.loc[(grammy_1['Name'] == 'Senorita')]

# check the won grammy name
grammy_1=grammy_1.rename(columns={'won grammy':'won_grammy'})

# only take the unique values, not sure why we have duplicated data
grammy_1 = grammy_1.drop_duplicates()
grammy_1.shape # (108343, 2)
grammy_1['won_grammy'].value_counts()


# read another dataset
billboard = pd.read_csv('billboardHot100_1999-2019.csv')
print(billboard.info())

# only select Lyrics and Name from the billboard dataset
subsetted = billboard[['Name','Lyrics']]
print(subsetted.info())
subsetted.isnull().sum()
subsetted.shape # (97225, 2)

subsetted = subsetted.drop_duplicates()
subsetted.shape # (7212, 2)


# merge the two datasets together
merged = pd.merge(subsetted,grammy_1,on='Name',how='inner')
print(merged.info())
merged.shape # (6126, 3)

# only take the unique values, not sure why we have duplicated data
merged_1 = merged.drop_duplicates()
merged_1.shape # (6126, 3)


# see the true/false distribution
distribution = merged_1['won_grammy'].value_counts()
sns.barplot(distribution.index,distribution)


# Replacing columns with f/t with 0/1
merged_1['won_grammy'] = pd.get_dummies(merged_1['won_grammy'],drop_first=True)

merged_1.info()

'''
     text preprocessing
'''

# return the wordnet object value corresponding to the POS tag

from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('averaged_perceptron_tagger')


def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove \n
    text = text.rstrip("\n")
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

# clean text data
merged_1["lyric_clean"] = merged_1["Lyrics"].apply(lambda x: clean_text(x))


# To clean textual data, we call our custom ‘clean_text’ function that performs several transformations:
# lower the text
# tokenize the text (split the text into words) and remove the punctuation
# remove useless words that contain numbers
# remove useless stop words like ‘the’, ‘a’ ,’this’ etc.
# Part-Of-Speech (POS) tagging: assign a tag to every word to define if it corresponds to a noun,
# a verb etc. using the WordNet lexical database
# lemmatize the text: transform every word into their root form (e.g. rooms -> room, slept -> sleep)


# add tf-idfs columns
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(merged_1["lyric_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = merged_1.index
merged_2 = pd.concat([merged_1, tfidf_df], axis=1)


merged_2['won_grammy'].value_counts()

merged_2.columns


t = merged_2[merged_2['won_grammy']== 1]
f = merged_2[merged_2['won_grammy']== 0]



'''
    wordclouds
'''

# wordcloud function

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    
# print wordcloud
show_wordcloud(t['lyric_clean'])

# feature selection
label = "won_grammy"
ignore_cols = [label, 'Name','Lyrics','lyric_clean']
features = [c for c in merged_2.columns if c not in ignore_cols]




X = merged_2.drop(['Name','Lyrics','lyric_clean','won_grammy'],axis=1)
y = merged_2['won_grammy']


# split the data into train and test
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(merged_2[features], merged_2[label],
                                                    test_size = 0.25, 
                                                    random_state = 42)


# train a random forest classifier
rf = RandomForestClassifier(n_estimators = 500, random_state = 42)
rf.fit(x_train, y_train)

feature_importances = pd.DataFrame(rf.feature_importances_, 
                                   index=X.columns,
                                   columns=['importance']).sort_values('importance', 
                                                                       ascending=False)
feature_importances

len(rf.feature_importances_)


# show feature importance
feature_importances_df2 = pd.DataFrame({"feature": features, 
                                       "importance": rf.feature_importances_}).sort_values("importance", ascending = False)
feature_importances_df2.head(20)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
y_pred3 = rf.predict(x_test)

matrix3 = pd.DataFrame(confusion_matrix(y_test,y_pred3,labels=[1,0]),
index=['actual:1','actual:0'],columns=['pred:1','pred:0'])

matrix3

print(classification_report(y_test,y_pred3))


rf.predict(["you fucking with some wet-ass pussy"])





'''
    NLP
'''
# text pre-processing:
# convert to lowercase, strip and remove puncutations
lyrics = merged_1['Lyrics']

X, y = merged_1.Lyrics,merged_1.won_grammy


from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import pickle
from nltk.corpus import stopwords



documents = []

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)


'''
    bag of words
'''

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=500, min_df=5, max_df=0.7,
                             stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()
# The fit_transform function of the CountVectorizer class converts text 
# documents into corresponding numeric features.


'''
    Finding TFIDF
'''
from sklearn.feature_extraction.text import TfidfTransformer

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=0)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, random_state=0)
rf.fit(X_train, y_train) 

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
y_pred = rf.predict(X_test)

matrix = pd.DataFrame(confusion_matrix(y_test,y_pred,labels=[1,0]),
index=['actual:1','actual:0'],columns=['pred:1','pred:0'])

matrix

print(classification_report(y_test,y_pred))


rf.predict(["you fucking with some wet-ass pussy"])










rf.feature_importance.plot_feature_importance(pyint_model, ascending=False, ax=ax)
    plt.show()
    
    
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index=X.columns,
                                   columns=['importance']).sort_values('importance', 
                                                                       ascending=False)
feature_importances = feature_importances.rename_axis('feature',axis="columns")
                                                                                
feature_importances = pd.DataFrame(feature_importances)
     
feature_importances.plot(kind='barh')









