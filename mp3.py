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
import torch.utils.data as Data

from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch


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

    # Clean data
    EMBEDDING_SIZE = 100
    MAX_LEN = 64

    # nn setup
    SEED = 1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_texts,train_sents = clean_data(training_data)
    test_texts,test_sents = clean_data(testing_data)

    train_labels = [data["label"] for data in training_data]
    
    tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=3)
    #model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=3)
    train_encodings = tokenizer(
        train_sents, truncation=True, padding=True, max_length=MAX_LEN)
    #test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LEN)

    train_dataset = TextDataset(train_encodings, train_labels)
    #test_dataset = TextDataset(valid_encodings, valid_labels)

    # training_args = TrainingArguments(
    #                     output_dir='./results',          # output directory
    #                     num_train_epochs=1,              # total number of training epochs
    #                     per_device_train_batch_size=16,  # batch size per device during training
    #                     #per_device_eval_batch_size=20,   # batch size for evaluation
    #                     warmup_steps=500,                # number of warmup steps for learning rate scheduler
    #                     weight_decay=0.01,               # strength of weight decay
    #                     #logging_dir='./logs',            # directory for storing logs
    #                     load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    #                     # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    #                     #logging_steps=200,               # log & save weights each logging_steps
    #                     #evaluation_strategy="steps",     # evaluate each `logging_steps`
    #                 )

    # trainer = Trainer(
    #                 model=model,                         # the instantiated Transformers model to be trained
    #                 args=training_args,                  # training arguments, defined above
    #                 train_dataset=train_dataset,         # training dataset
    #                 #eval_dataset=valid_dataset,          # evaluation dataset
    #                 compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    #             )
    #trainer.train()
    epochs = 2
    batch_size = 32


    # # classify train_texts and balance train_texts
    class_count = np.array([train_labels.count(0), train_labels.count(1), train_labels.count(2)])
    weight = 1. / class_count
    samples_weight = torch.from_numpy(
        np.array([weight[t] for t in train_labels])).double()
    sampler = Data.WeightedRandomSampler(samples_weight, len(samples_weight))
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    
    for epoch in range(epochs):
        counter = 0
        epoch_loss = 0.0
        epoch_acc = 0.0
        for batch in train_loader:
            counter += batch_size
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            predictions = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = predictions[0]
            loss.backward()
            optimizer.step()

            acc = accuracy(predictions[1], labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            print("Epoch: {}/{}".format(epoch+1, epochs),
                  "Step: {}".format(counter),
                  "Loss: {:.6f}".format(loss.item()),
                  "Accuracy:", acc.item(),
                  "epoch_loss_sum:", epoch_loss, 
                  "epoch_acc_sum:", epoch_acc)

    


    

    result = []
    #lstm_model.eval()
    model.eval()
    with torch.no_grad():
        for sent in test_sents:
            #label = torch.argmax(lstm_model(sent.view(1, -1)), dim=1)
            inputs = tokenizer(sent, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            # perform inference to our model
            outputs = model(inputs["input_ids"], inputs["attention_mask"])  
            # get output probabilities by doing softmax
            probs = outputs[0].softmax(1)
            # executing argmax function to get the candidate label
            result.append(probs.argmax())
    return result


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)





def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


def accuracy(preds, y):
    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    #acc = correct.float() / y.shape[0]
    return correct


def process_data(texts, word_to_ix):
    max_l = 50

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))
    features = torch.tensor(
        [pad([word_to_ix[w] for w in sent]) for sent in texts])
    return features


def clean_data(dataset):
    texts = []
    sentences = []
    for data in dataset:
        cleaned_data = decontracted(data["text"].lower().encode(
            'ascii', errors='ignore').decode('ascii'))
        tokenwords = nltk.word_tokenize(cleaned_data)
        # tokenwords.append(data["speaker"].lower())
        texts.append(tokenwords)
        sentences.append(cleaned_data)
    return texts,sentences


def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
