# Starter code for CS 165B MP3
import random
import numpy as np
import string
import nltk
import tqdm
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

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


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
    EMBEDDING_SIZE = 70
    train_texts,train_sentences = clean_data(training_data)
    test_texts,test_sentences = clean_data(testing_data)
    total_texts = train_texts + test_texts
    labels = [data["label"] for data in training_data]
    w2v_model = Word2Vec(sentences=total_texts, vector_size=EMBEDDING_SIZE, window=5, min_count=1, workers=4)
    w2v_words = list(w2v_model.wv.index_to_key)

    
    documents = []
    for i in range(len(train_texts)):
        documents.append(TaggedDocument(train_texts[i], [labels[i]]))
    
    
    model = Doc2Vec(documents, vector_size=EMBEDDING_SIZE, window=5, min_count=1, workers=4, epochs=20)
    # model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=50)

    train_vectors = []
    for sent in train_texts:
        train_vectors.append(model.infer_vector(sent))

    test_vectors = []
    for sent in test_texts:
        test_vectors.append(model.infer_vector(sent))

    #Implement support vector machine
    #param_grid = {'C': [400,500,600], 'kernel': ['poly', 'rbf'], 'gamma': [0.0, 0.1, 0.01,]}
    # param_grid = {'C': [500], 'kernel': ['poly', 'rbf'], 'gamma': [0.01]}
    #clf = GridSearchCV(svm.SVC(degree=5, max_iter=10000), param_grid= param_grid, refit=True,)
    
    #clf = svm.SVC(C=100,kernel='rbf')
    C=600
    GAMMA = 0.01
    DEGREE = 3
    MAX_ITER = 30000
    clf = svm.SVC(kernel='rbf',max_iter=MAX_ITER, C = C, gamma=GAMMA, degree=DEGREE)
    clf.fit(train_vectors, labels)

    #print(clf.best_params_)
    #result = clf.best_estimator_.predict(test_vectors)   

    print("EMBEDDING_SIZE=", EMBEDDING_SIZE, ", C=", C, ", gamma=", GAMMA, ", degree=", DEGREE, ", max_itr=", MAX_ITER)
    result = clf.predict(test_vectors)

    

    return result






def clean_data(dataset):
    texts = []
    sentences = []
    for data in dataset:
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

