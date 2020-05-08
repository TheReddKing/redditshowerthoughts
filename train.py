import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(device)


vocabfile = open('vocab_showerthoughts.txt', "r")
dictionary = []
for w in vocabfile.readlines():
    if (len(w.strip()) == 0):
        dictionary.append(" ")
    else:
        dictionary.append(w.strip())

vocab_to_id = {}
id_to_vocab = {}
for i in range(len(dictionary)):
    vocab_to_id[dictionary[i]] = i
    id_to_vocab[i] = dictionary[i]

data = open('clean_showerthoughts.txt','r')
all_data = [[int(i) for i in d.strip().split(" ")] for d in data.readlines()]

# 10%
split = len(all_data) // 10

corpus_valid = all_data[:split]
corpus_train = all_data[split:]

hidden_size = 30
num_words = len(dictionary)
batch_size = 1
num_layers = 2
lr = 0.05
num_epochs = 10 #Be aware of over-fitting!
loss_fn = nn.CrossEntropyLoss().to(device)
dropout = 0.25

class SimpleLSTM(nn.Module):
    def __init__(self, num_words, num_layers, hidden_size, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_words, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size*2, num_words)
    
    def forward(self, x, prev_state):
        conv = self.embedding(x)
        conv = self.dropout(conv)
        output, state = self.lstm(conv)
        logits = self.dense(output)
        return logits, state
    
    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device))

classifier = SimpleLSTM(num_words, num_layers, hidden_size, dropout).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

def one_hot_encoding(ind, size):
    one_hot = torch.zeros([1,size], dtype=torch.long).to(device)
    one_hot[:,ind] = 1
    return one_hot

classifier = classifier.to(device)
classifier.train()
for epoch in range(num_epochs):
    total_loss = 0
    total_seen = 0.001
    classifier.train()
    for i in range(len(corpus_train)):
        if i % 100 == 0:
            print('Epoch {} Batch {} LOSS: {}'.format(epoch, i, total_loss / total_seen))
            total_loss = 0
            total_seen = 0.001
        data = torch.tensor(corpus_train[i]).to(device)
        state_h, state_c = classifier.zero_state(batch_size)
        # TODO IMPLEMENT BATCH_SIZE
        for j in range(len(data) - 1):
            classifier.zero_grad()
            node = (data[j:j+1][None, :])
            logits, (state_h, state_c) = classifier(node, (state_h, state_c))
            loss = loss_fn(logits[0], data[j+1:j+2])
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().data.numpy().sum()
            total_seen += 1
    print('Epoch {}, train loss={}'.format(epoch, total_loss / total_seen))
    torch.save(classifier, '/content/drive/My Drive/000Data/ShowerThoughts/model_' + str(epoch) + '.bin')

    total_loss = 0
    total_seen = 0
    classifier.eval()
    for i in range(len(corpus_valid)):
        data = torch.tensor(corpus_train[i]).to(device)
        state_h, state_c = classifier.zero_state(batch_size)
        
        for i in range(len(data) - 1):
            node = (data[i:i+1][None, :])
            output, (state_h, state_c) = classifier(node, (state_h, state_c))
            loss = loss_fn(output[0], data[i+1:i+2])
            total_loss += loss.cpu().data.numpy().sum()
            total_seen += 1
        # --------- Your code ends --------- #
    print('Epoch {}, valid loss={}'.format(epoch, total_loss / total_seen))