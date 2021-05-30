# Starter code for CS 165B MP3
import random
import numpy as np

import nltk

import re
from gensim.models import Word2Vec
from numpy import dot
from numpy.linalg import norm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sklearn
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def run_train_test(training_data, testing_data):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: List[{"text": the utterance,\
                             "label": the label, can be 0(negative), 1(neutral),or 2(positive),\
                             "speaker": the name of the speaker,\
                             "dialogue_id": the id of dialogue,\
                             "utterance_id": the id of the utterance}]
        testing_data: the same as training_data with "label" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1, 2], k=len(testing_data))
    """
    #Get word2vec
    EMBEDDING_SIZE = 100
    train_texts,train_sentences = clean_data(training_data)
    test_texts,test_sentences = clean_data(testing_data)
    total_texts = train_texts + test_texts
    labels = [data["label"] for data in training_data]
    w2v_model = Word2Vec(sentences=total_texts, vector_size=EMBEDDING_SIZE, window=5, min_count=1, workers=4, epochs=20)
    w2v_words = list(w2v_model.wv.index_to_key)

    #Calculate TFIDF weighted Word2Vec for training features
    tfidf_model = sklearn.feature_extraction.text.TfidfVectorizer()
    tfidf_model.fit(train_sentences+test_sentences)
    dictionary = dict(zip(tfidf_model.get_feature_names(),list(tfidf_model.idf_)))
    tfidf_feat = tfidf_model.get_feature_names()
    tfidf_w2v_features_train = []; 
    for sent in train_texts: 
        sent_vec = np.zeros(EMBEDDING_SIZE) 
        weight_sum =0; 
        for word in sent: 
            if word in w2v_words and word in tfidf_feat:
                vec = w2v_model.wv[word]
                tf_idf = dictionary[word]*(sent.count(word)/len(sent))
                sent_vec += (vec * tf_idf)
                weight_sum += tf_idf
        if weight_sum != 0:
            sent_vec /= weight_sum
        tfidf_w2v_features_train.append(sent_vec)

    features = []
    for sentence in train_texts:
        features.append(np.zeros(EMBEDDING_SIZE))
        for word in sentence:
            features[-1] += w2v_model.wv[word]
        features[-1] /= len(sentence)
    
    

    

    #Calculate TFIDF weighted Word2Vec for testing features
    # tfidf_model_test = sklearn.feature_extraction.text.TfidfVectorizer()
    # tfidf_model_test.fit(test_sentences)
    # dictionary_test = dict(zip(tfidf_model_test.get_feature_names(),list(tfidf_model_test.idf_)))
    # tfidf_feat_test = tfidf_model_test.get_feature_names()
    tfidf_w2v_features_test = []; 
    for sent in test_texts: 
        sent_vec = np.zeros(EMBEDDING_SIZE) 
        weight_sum =0; 
        for word in sent: 
            if word in w2v_words and word in tfidf_feat:
                vec = w2v_model.wv[word]
                tf_idf = dictionary[word]*(sent.count(word)/len(sent))
                sent_vec += (vec * tf_idf)
                weight_sum += tf_idf
        if weight_sum != 0:
            sent_vec /= weight_sum
        tfidf_w2v_features_test.append(sent_vec)

    test_features = []
    for sentence in test_texts:
        test_features.append(np.zeros(EMBEDDING_SIZE))
        for word in sentence:
            test_features[-1] += w2v_model.wv[word]
        test_features[-1] /= len(sentence)


    #Implement support vector machine
    #param_grid = {'C': [400,500,600], 'kernel': ['poly', 'rbf'], 'gamma': [0.0, 0.1, 0.01,]}
    # param_grid = {'C': [500], 'kernel': ['poly', 'rbf'], 'gamma': [0.01]}
    #clf = GridSearchCV(svm.SVC(degree=5, max_iter=10000), param_grid= param_grid, refit=True,)
    C=500
    GAMMA = 0.01
    DEGREE = 3
    MAX_ITER = 25000
    clf = svm.SVC(kernel='rbf',max_iter=MAX_ITER, C = C, gamma=GAMMA, degree=DEGREE)
    #clf = svm.SVC(kernel='rbf',max_iter=10000, C = 500, gamma=0.1, degree=5)
    #clf.fit(tfidf_w2v_features_train, labels)
    clf.fit(features, labels)

    #print(clf.best_params_)
    #result = clf.best_estimator_.predict(test_features)   
    #result = clf.best_estimator_.predict(tfidf_w2v_features_test)
    result = clf.predict(test_features)
    #result = clf.predict(tfidf_w2v_features_test)
    print("EMBEDDING_SIZE=", EMBEDDING_SIZE, ", C=", C, ", gamma=", GAMMA, ", degree=", DEGREE, ", max_itr=", MAX_ITER)
    return result






def clean_data(training_data):
    texts = []
    sentences = []
    for data in training_data:
        cleaned_data = decontracted(data["text"].lower().encode('ascii',errors='ignore').decode('ascii'))#
        #cleaned_data = re.sub("\S*\d\S*", "", cleaned_data).strip()
        tokenwords = nltk.word_tokenize(cleaned_data) 
        tokenwords.append(data["speaker"].lower())
        texts.append(tokenwords)
        sentences.append(cleaned_data)

    return texts,sentences

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

