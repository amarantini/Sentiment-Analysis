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
    


    #Clean data
    EMBEDDING_SIZE = 100

    train_texts,train_sentences = clean_data(training_data) #
    test_texts,test_sentences = clean_data(testing_data) #
    total_texts = train_texts + test_texts
    labels = [data["label"] for data in training_data]

    #classify train_texts and balance train_texts 
    class_count = np.array([labels.count(0),labels.count(1),labels.count(2)])
    weight = 1. / class_count
    samples_weight = torch.from_numpy(np.array([weight[t] for t in labels])).double()
    sampler = Data.WeightedRandomSampler(samples_weight, len(samples_weight))
    # classified_train_texts={0:[],1:[],2:[]}
    # for i,label in enumerate(labels):
    #     classified_train_texts[label].append([train_texts[i],label])
    
    # min_class = min(len(classified_train_texts[0]),len(classified_train_texts[1]),len(classified_train_texts[2]))
    # balanced_train_texts_labels = []
    # balanced_train_texts_labels += random.sample(classified_train_texts[0],min_class)
    # balanced_train_texts_labels += random.sample(classified_train_texts[1],min_class)
    # balanced_train_texts_labels += random.sample(classified_train_texts[2],min_class)
    # random.shuffle(balanced_train_texts_labels)
    # balanced_train_labels = [item[1] for item in balanced_train_texts_labels]
    # balanced_train_texts = [item[0] for item in balanced_train_texts_labels]
    #Get word2vec
    w2v_model = Word2Vec(sentences=total_texts, vector_size=EMBEDDING_SIZE, window=5, min_count=1, workers=4)
    w2v_words = list(w2v_model.wv.index_to_key)
    word_to_ix = {}

    PAD_IDX = 0
    word_to_ix['<pad>']=PAD_IDX
    for i,word in enumerate(w2v_words):
        word_to_ix[word]=i+1

    

    w2v_embed = [0]*len(word_to_ix)
    for word, i in word_to_ix.items():
        if word != '<pad>':
            w2v_embed[i]=w2v_model.wv[word]
        else:
            w2v_embed[i]=[0]*EMBEDDING_SIZE
    
    #parameters
    INPUT_DIM = len(word_to_ix)
    HIDDEN_DIM = 256
    OUTPUT_DIM = 3
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    
    #nn setup
    SEED = 1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 16
    processed_data = process_data(train_texts, word_to_ix)#balanced_train_texts
    train_set = Data.TensorDataset(processed_data,torch.tensor(labels).long())#balanced_train_labels
    train_iter = Data.DataLoader(train_set, batch_size, sampler=sampler)#shuffle=True,



    lstm_model = LSTM(vocab_size=INPUT_DIM, 
                    embedding_dim=EMBEDDING_SIZE, 
                    hidden_dim=HIDDEN_DIM, 
                    output_dim=OUTPUT_DIM, 
                    n_layers=N_LAYERS, 
                    bidirectional=BIDIRECTIONAL, 
                    dropout=DROPOUT)
    loss_function = nn.CrossEntropyLoss()#NLLLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001) #optim.Adam(model.parameters())


    lstm_model.embedding.weight.data.copy_(torch.tensor(w2v_embed))
    #lstm_model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)
    lstm_model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
    
    lstm_model = lstm_model.to(device)
    loss_function = loss_function.to(device)
    
    best_valid_loss = float('inf')
    lstm_model.train()
    epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        counter = 0
        for batch in train_iter:
            counter += batch_size
            optimizer.zero_grad()
            
            inputs, labels = batch

            predictions = lstm_model(inputs)#.squeeze(1)

            loss = loss_function(predictions, labels)
            
            acc = accuracy(predictions, labels)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            # print("Epoch: {}/{}".format(epoch+1, epochs),
            #       "Step: {}".format(counter),
            #       "Loss: {:.6f}".format(loss.item()),
            #       "Accuracy:", acc.item(),
            #       "epoch_loss_sum:", epoch_loss, 
            #       "epoch_acc_sum:", epoch_acc)

        # if loss < best_valid_loss: 
        #     best_valid_loss = loss
        #     torch.save(lstm_model.state_dict(), 'word2vec-lstm-model-balanced_b4.pt')

    processed_test_data = process_data(test_texts, word_to_ix)
    result=[]
    lstm_model.eval()
    with torch.no_grad():
        for sent in processed_test_data:
            label = torch.argmax(lstm_model(sent.view(1, -1)), dim=1)
            result.append(label)
    return result




class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_dim,n_layers,bidirectional, dropout, pad_idx=0):
        super().__init__() #LSTM, BiRNN, self

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_dim,
                            num_layers=n_layers, 
                            bidirectional=bidirectional,
                            dropout=dropout) 
        self.linear_1 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim*2, hidden_dim*1)
        self.predictor = nn.Linear(hidden_dim*1, output_dim) #hidden_dim*4, output_dim
        self.dropout = nn.Dropout(dropout)


    def forward(self, input):
        embeds = self.dropout(self.embedding(input.permute(1, 0)))#self.dropout(self.embeddings(sentence)) #[sent len, batch size, emb dim]
        lstm_out, (hidden, cell) = self.lstm(embeds)
        hidden = self.dropout(torch.cat((lstm_out[0], lstm_out[-1]), -1))
        hidden = self.linear_1(hidden)
        hidden = self.relu(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.relu(hidden)
        #hidden = torch.cat((lstm_out[0], lstm_out[-1]), -1)
        #hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)#self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        outs = self.predictor(hidden) #self.predictor(hidden) 
        
        #lstm_out = [sent len, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        #hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)) # [batch size, hid dim * num directions]
        return outs
        #word_label = self.predictor(lstm_out.view(len(sentence), -1))
        #word_label_norm = F.log_softmax(word_label, dim=1)
        #sentence_label = nn.ReLU(word_label_norm)

def accuracy(preds, y):
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    #acc = correct.float() / y.shape[0]
    return correct

def process_data(texts,word_to_ix):
    max_l = 50
    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))
    features = torch.tensor([pad([word_to_ix[w] for w in sent]) for sent in texts])
    return features


def clean_data(dataset):
    texts = []
    sentences = []
    for data in dataset:
        cleaned_data = decontracted(data["text"].lower().encode('ascii',errors='ignore').decode('ascii'))#
        #cleaned_data = re.sub("\S*\d\S*", "", cleaned_data).strip()
        tokenwords = nltk.word_tokenize(cleaned_data) 
        #tokenwords.append(data["speaker"].lower())
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



