import pandas as pd
import os, glob
from parsivar import Tokenizer, Normalizer, FindStems
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.utils import shuffle


import requests
from string import punctuation

from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_curve, roc_auc_score, accuracy_score


class GettingRidOfSpammers:

    def __init__(self,
                 directory_spam,
                 directory_not_spam):

        self.directory_spam = directory_spam
        self.directory_not_spam = directory_not_spam

        self._preprocess()


    def _preprocess(self):

        self._read_spams()
        self._read_not_spams()

        data = [self.df_spam, self.df_not_spam]
        data = pd.concat(data)
        self.data = shuffle(data)

        vectorizer = TfidfVectorizer(max_features = 300,
                                     preprocessor = self._normalize_text,
                                     use_idf = True).fit(self.data.email)

        print(vectorizer.get_feature_names())

        self.x = vectorizer.transform(self.data.email)
        self.y = self.data.spam

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                                            random_state = 313, test_size = 0.2)


    def _read_spams(self):
        texts = []
        text_file = []

        # iterate over files in
        # that directory
        for filename in os.listdir(self.directory_spam):
            f = os.path.join(self.directory_spam, filename)
            # checking if it is a file
            if os.path.isfile(f):
                file = open(f, encoding = "utf8").read()
                for ind, line in enumerate(file.split('\n\n')):
                    text = line.replace('\n', '')  # delete all the single '\n'
                    text = line.replace('\n\n', '')  # delete all the '\n\n'
                    texts.append(line)
                listToStr = ' '.join(map(str, texts))
                d = {'email': listToStr, 'file': f.split('\\')[-1]}
                text_file.append(d)

        self.df_spam = pd.DataFrame({'text': text_file})
        self.df_spam = pd.json_normalize(self.df_spam['text'])
        self.df_spam['spam'] = 1

    def _read_not_spams(self):
        textstr = []
        text_filetr = []

        # iterate over files in
        # that directory
        for filename in os.listdir(self.directory_not_spam):
            fr = os.path.join(self.directory_not_spam, filename)
            # checking if it is a file
            if os.path.isfile(fr):
                filer = open(fr, encoding = "utf8").read()
                for ind, line in enumerate(filer.split('\n\n')):
                    text = line.replace('\n', '')  # delete all the single '\n'
                    text = line.replace('\n\n', '')  # delete all the '\n\n'
                    textstr.append(line)
                listToStrr = ' '.join(map(str, textstr))
                dr = {'email': listToStrr, 'file': fr.split('\\')[-1]}
                text_filetr.append(dr)

        self.df_not_spam = pd.DataFrame({'text': text_filetr})
        self.df_not_spam = pd.json_normalize(self.df_not_spam['text'])
        self.df_not_spam['spam'] = 0

    def _normalize_text(self, text):

        tokenizer = Tokenizer()
        normalizer = Normalizer()
        stemmer = FindStems()
        my_punctuation = punctuation + '،"؛«»)\('

        tokens = tokenizer.tokenize_words(text)
        tokens = [stemmer.convert_to_stem(word).split('&')[0] for word in tokens]
        text = ' '.join([word for word in tokens if word not in list(my_punctuation)])

        return text

    def train_model(self):
        # you can use any kind of models here! It's just a sample!!!

        model = RandomForestClassifier(n_jobs = -1,
                                       n_estimators = 45,
                                       random_state = 313,
                                       class_weight = 'balanced')
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)

        output = pd.DataFrame()
        output['filename'] = self.data['file']
        output['prediction'] = y_pred

        return output



if __name__ == '__main__':

    directory_spam = 'spam_training'
    directory_not_spam = 'ok_training'

    bored = GettingRidOfSpammers(directory_spam= directory_spam,
                                 directory_not_spam = directory_not_spam)
    output = bored.train_model()
    output.to_csv('./output.csv')


