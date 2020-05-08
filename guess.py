import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(device)

vocabfile = open('vocab_showerthoughts_10k.txt', "r")
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

# LET's process the data now
vocab_length = sorted(dictionary, key=lambda x: -len(x))

# hidden_size = 30
# num_words = len(dictionary)
# batch_size = 1
# num_layers = 2
# lr = 0.05
# num_epochs = 10 #Be aware of over-fitting!
# loss_fn = nn.CrossEntropyLoss().to(device)
# dropout = 0.25
hidden_size = 30
num_words = len(dictionary)
batch_size = 1
num_layers = 1
lr = 0.001
num_epochs = 10 #Be aware of over-fitting!
loss_fn = nn.CrossEntropyLoss().to(device)
dropout = 0.05
class SimpleLSTM(nn.Module):
    def __init__(self, num_words, num_layers, hidden_size, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.layers = num_layers
        self.embedding = nn.Embedding(num_words, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=False)
        self.dense = nn.Linear(hidden_size, num_words)
    
    def forward(self, x, prev_state):
        conv = self.embedding(x)
        output, state = self.lstm(conv, prev_state)
        logits = self.dense(output)
        return logits, state
    
    def zero_state(self, batch_size):
        return (torch.zeros(self.layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.layers, batch_size, self.hidden_size).to(device))

def one_hot_encoding(ind, size):
    one_hot = torch.zeros([1,size], dtype=torch.long).to(device)
    one_hot[:,ind] = 1
    return one_hot
# class SimpleLSTM(nn.Module):
#     def __init__(self, num_words, num_layers, hidden_size, dropout=0.2):
#         super(SimpleLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(num_words, hidden_size)
#         self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
#                            batch_first=True, dropout=dropout, bidirectional=False)
#         self.dense = nn.Linear(hidden_size, num_words)
    
#     def forward(self, x, prev_state):
#         conv = self.embedding(x)
#         output, state = self.lstm(conv)
#         logits = self.dense(output)
#         return logits, state
    
#     def zero_state(self, batch_size):
#         return (torch.zeros(1, batch_size, self.hidden_size).to(device),
#                 torch.zeros(1, batch_size, self.hidden_size).to(device))

# classifier_m = SimpleLSTM(num_words, num_layers, hidden_size, dropout).to(device)
# classifier = torch.load("model_8.bin", map_location=torch.device(device))
classifier = torch.load("model_0_40_2layers_0.05.bin", map_location=torch.device(device))

import enchant
d = enchant.Dict("en_US")

def nextWord(current_word, state, node, id):
    if (len(current_word) > 10):
        return []
    if (len(current_word) > 1 and current_word[-1] == ' ' and d.check(current_word.strip())):
        # END OF WORD
        return [(current_word.strip(), state, node[0][0].item(), False)]
    node[0][0] = id
    output, state = classifier(node, state)
    ids = torch.topk(output, k=3)[1][0,0]
    allWords = []
    for i, id in enumerate(ids):
        if (id.item() == 2): # Space
            node[0][0] = id
            allWords.extend(nextWord(current_word + id_to_vocab[id.item()], state, node, id))
            # break
        if (id.item() != 1):
            node[0][0] = id
            allWords.extend(nextWord(current_word + id_to_vocab[id.item()], state, node, id))
        elif (d.check(current_word.strip())):
            allWords.append((current_word.strip(), state, node[0][0].item(), True))
    return allWords

while True:
    sentence = input("Enter sentence: ")
    l = sentence
    tokens = []
    tokens.append(vocab_to_id["[START]"])
    while (len(sentence) > 0):
        if (sentence[0] == ' '):
            tokens.append(vocab_to_id[' '])
            sentence = sentence[1:]
        for w in vocab_length:
            if sentence.startswith(w):
                tokens.append(vocab_to_id[w])
                sentence = sentence[len(w):]
                break
    data = torch.tensor(tokens).to(device)
    print(tokens)
    state = classifier.zero_state(batch_size)
    for i in range(len(data)):
        node = (data[i:i+1][None, :])
        output, state = classifier(node, state)
        ids = torch.topk(output, k=3)[1][0,0]
        print(id_to_vocab[data[i].item()], [id_to_vocab[id.item()] for id in ids])

    if (ids[1].item() == 1):
        node[0][0] = ids[0].item()
    else:
        node[0][0] = ids[random.randint(0,1)].item()
    
    res = []
    mm = 0
    while mm < 10:
        mm += 1
        if (ids[0].item() == 1):
            break

        res = nextWord('', state, node, node[0][0])
        print(res[0][0])
        node[0][0] = res[0][2]
    # print(l + "".join(res))