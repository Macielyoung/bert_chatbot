# coding:utf-8

import torch
import re
import os
import unicodedata

from config import MAX_LENGTH, save_dir

SOS_token = 0
EOS_token = 1
PAD_token = 2

class Voc:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"PAD"}
        self.n_words = 3  # Count SOS and EOS
        self.loc2index = {}   # 位置到索引号
        self.index2loc = {}   # 索引号到位置
        self.loc2word = {}    # 位置到单词
        self.count2word = {0: "SOS", 1: "EOS", 2: "PAD"}   # 索引号到单词
        self.loc_count = 3    # 单词总数
        self.special = ["SOS", "EOS", "PAD"]

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_line_word(self, sentence):
        # print(self.loc_count)
        # print(sentence)
        row_index = sentence[0]
        for num, value in enumerate(sentence[1].split(" ")):
            if value not in self.special:
                self.loc2word[(row_index, num)] = value
                self.loc2index[(row_index, num)] = self.loc_count
                self.index2loc[self.loc_count] = (row_index, num)
                self.count2word[self.loc_count] = value
                self.loc_count += 1
            else:
                if value == "SOS":
                    self.loc2index[(row_index, num)] = 0
                elif value == "EOS":
                    self.loc2index[(row_index, num)] = 1
                else:
                    self.loc2index[(row_index, num)] = 2

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readVocs(corpus, corpus_name):
    print("Reading lines...")

    # combine every two lines into pairs and normalize
    with open(corpus) as f:
        content = f.readlines()
    # import gzip
    # content = gzip.open(corpus, 'rt')
    lines = [x.lower().strip() for x in content]
    pairs = []
    input, output = "", ""
    for row, line in enumerate(lines):
        if row % 2 == 0:
            input = (row, line)
        else:
            output = (row, line)
            pairs.append([input, output])

    # it = iter(lines)
    # # pairs = [[normalizeString(x), normalizeString(next(it))] for x in it]
    # pairs = [[x, next(it)] for x in it]
    # print(pairs)

    voc = Voc(corpus_name)
    return voc, pairs

def filterPair(p):
    # input sequences need to preserve the last word for EOS_token
    return len(p[0].split(' ')) <= MAX_LENGTH and \
        len(p[1].split(' ')) <= MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(corpus, corpus_name):
    voc, pairs = readVocs(corpus, corpus_name)
    # print(MAX_LENGTH)
    # print(len(pairs))
    print("Read {!s} sentence pairs".format(len(pairs)))
    # pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        # voc.addSentence(pair[0])
        # voc.addSentence(pair[1])
        voc.add_line_word(pair[0])
        voc.add_line_word(pair[1])
    print("Counted words:", voc.loc_count)
    directory = os.path.join(save_dir, 'training_data', corpus_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(voc, os.path.join(directory, '{!s}.tar'.format('voc')))
    torch.save(pairs, os.path.join(directory, '{!s}.tar'.format('pairs')))
    return voc, pairs

def loadPrepareData(corpus):
    corpus_name = corpus.split('/')[-1].split('.')[0]
    # try:
    #     print("Start loading training data ...")
    #     voc = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'voc.tar'))
    #     pairs = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'pairs.tar'))
    # except FileNotFoundError:
    print("Saved data not found, start preparing training data ...")
    voc, pairs = prepareData(corpus, corpus_name)
    return voc, pairs

if __name__ == "__main__":
    corpus = "data/conversations.txt"
    voc, pairs = loadPrepareData(corpus)
    print(pairs[0])
    word_num = []
    # for pair in pairs:
    #     query, answer = pair[0], pair[1]
    #     query_words = query[1].split(" ")
    #     answer_words = answer[1].split(" ")
    #     num1 = len(query_words)
    #     num2 = len(answer_words)
    #     if num1 == 25 or num2 == 25:
    #         print("25 words: ", pair)
    #     if num1 == 23 or num2 == 23:
    #         print("23 words: ", pair)
    #     word_num.append(num1)
    #     word_num.append(num2)
    # unique_num = set(word_num)
    # all_num = sum(word_num)
    # print(unique_num, all_num)

    print(voc.loc2word)
    print(voc.loc2index)
    print(voc.count2word)
