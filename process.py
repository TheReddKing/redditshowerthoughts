import nltk
from nltk.tokenize import word_tokenize

freq = {}

create_vocab = True

if create_vocab:
    # import the words
    data = open('data_showerthoughts.txt','r')

    for l in data.readlines():
        id, title = l.split("|")
        words = title.strip().split(" ")
        for w in words:
            if w not in freq:
                freq[w] = 0
            freq[w] += 1
    data.close()
    import operator
    sorted_d = sorted(freq.items(), key=operator.itemgetter(1))[::-1]
    print(len(sorted_d))
    dictionary = []
    dictionary.append("[START]")
    dictionary.append("[STOP]")
    dictionary.append(" ")
    dictionary.extend(list("qwertyuiopasdfghjklzxcvbnm1234567890"))

    # 300 different words
    for i in range(3000):
        v = sorted_d[i][0]
        if len(v) > 1:
            dictionary.append(v)

    print(dictionary)

    vocabfile = open('vocab_showerthoughts.txt', "w")
    for w in dictionary:
        vocabfile.write(w + '\n')
else:
    # Load vocab
    vocabfile = open('vocab_showerthoughts.txt', "r")
    dictionary = []
    for w in vocabfile.readlines():
        if (len(w.strip()) == 0):
            dictionary.append(" ")
        else:
            dictionary.append(w.strip())


vocab_to_id = {}
for i in range(len(dictionary)):
    vocab_to_id[dictionary[i]] = i

# LET's process the data now
vocab_length = sorted(dictionary, key=lambda x: -len(x))

data = open('data_showerthoughts.txt','r')
out = open('clean_showerthoughts.txt','w')
for l in data.readlines():
    sentence = l.strip().split("|")[1   ]
    tokens = []
    tokens.append(str(vocab_to_id["[START]"]))
    while (len(sentence) > 0):
        if (sentence[0] == ' '):
            tokens.append(str(vocab_to_id[' ']))
            sentence = sentence[1:]
        for w in vocab_length:
            if sentence.startswith(w):
                tokens.append(str(vocab_to_id[w]))
                sentence = sentence[len(w):]
                break
    tokens.append(str(vocab_to_id["[STOP]"]))
    out.write(" ".join(tokens) + "\n")
    # okay now let's replace some with words in general
out.close()
data.close()