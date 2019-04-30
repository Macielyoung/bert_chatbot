# coding:utf-8

import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import itertools
import random
import math
import os
from tqdm import tqdm
from load import loadPrepareData
from load import SOS_token, EOS_token, PAD_token
from model import EncoderRNN, LuongAttnDecoderRNN
from config import MAX_LENGTH, teacher_forcing_ratio, save_dir
from vectorization import json_sentence, embedding

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

cudnn.benchmark = True
#############################################
# generate file name for saving parameters
#############################################
def filename(reverse, obj):
	filename = ''
	if reverse:
		filename += 'reverse_'
	filename += obj
	return filename

#############################################
# Prepare Training Data
#############################################
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

# batch_first: true -> false, i.e. shape: seq_len * batch
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def sentence_to_index(row_sentence_pair, voc):
    row_index = row_sentence_pair[0]
    return [voc.loc2index[(row_index, row_num)] for row_num in range(len(row_sentence_pair[1].split(" ")))] + [EOS_token]

def bert_embedding(row, sentence, embedding_dict, voc):
    content = sentence.split(" ")
    json_sen = json_sentence(embedding[row])
    origin_sentence, embedded = json_sen.get_embedding(content)
    for num in range(len(content)):
        index = voc.loc2index[(row, num)]
        embedding_dict[index] = embedded[num]

def nn_embedding(embedding_dict, hidden_size):
    nn_embedding = nn.Embedding(3, hidden_size)
    particular_token = [SOS_token, PAD_token, EOS_token]
    for token in particular_token:
        tensor = torch.LongTensor([token])
        token_embedding = nn_embedding(tensor)[0].tolist()
        embedding_dict[token] = token_embedding

# 查找每个位置单词的Bert Embedding字典
def concate_embedding(pairs, voc, hidden_size):
    embedding_dict = {}
    # bert embedding for words
    for pair in pairs:
        question, answer = pair[0], pair[1]
        qrow, qsentence = question[0], question[1]
        get_embedding(qrow, qsentence, embedding_dict, voc)
        arow, asentence = answer[0], answer[1]
        get_embedding(arow, asentence, embedding_dict, voc)

    # nn embedding for particular words (SOS, PAD and EOS)
    nn_embedding(embedding_dict)
    return embedding_dict

# convert to index, add EOS
# return input pack_padded_sequence
def inputVar(input_batch, voc):
    indexes_batch = [sentence_to_index(pair, voc) for pair in input_batch]
    # print("input_batch", indexes_batch)
    lengths = [len(indexes) for indexes in indexes_batch]
    padList = zeroPadding(indexes_batch)
    # print("padding:", padList)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# convert to index, add EOS, zero padding
# return output variable, mask, max length of the sentences in batch
def outputVar(output_batch, voc):
    indexes_batch = [sentence_to_index(pair, voc) for pair in output_batch]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, max_target_len

# pair_batch is a list of (input, output) with length batch_size
# sort list of (input, output) pairs by input length, reverse input
# return input, lengths for pack_padded_sequence, output_variable, mask
def batch2TrainData(pair_batch, voc, reverse):
    if reverse:
        pair_batch = [pair[::-1] for pair in pair_batch]
    pair_batch.sort(key=lambda x: len(x[0][1].split(" ")), reverse=True)
    # print("pair_batch:", pair_batch[:5])
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    # print(input_batch)
    input_vec, lengths = inputVar(input_batch, voc)
    # print(input_vec.shape)
    output_vec, max_target_len = outputVar(output_batch, voc)
    # print(output_vec.shape)
    return input_vec, lengths, output_vec, max_target_len

# def get_sos(target_variable, batch_size):
#     start_sentences = []
#     for line in target_variable:
#         # print(line[0].shape)
#         start_sentences.append(line[0])
#     # print(start_sentences)
#     start_tensor = torch.cat(start_sentences, -1).view(batch_size, -1)
#     return start_tensor

#############################################
# Training
#############################################

def train(input_variable, lengths, target_variable, max_target_len, encoder, decoder, embedding_dict,
          encoder_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # print("target:", target_variable.size())

    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)

    # mask = mask.to(device)

    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, embedding_dict, lengths, None)
    # print(encoder_outputs.shape)

    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    # print("decoder:", decoder_input.size())

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Run through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, embedding_dict, decoder_hidden, encoder_outputs
            )
            decoder_input = target_variable[t]  # Next input is current target
            decoder_input = decoder_input.float()
            loss += F.cross_entropy(decoder_output, target_variable[t], ignore_index=EOS_token)
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, embedding_dict, decoder_hidden, encoder_outputs
            )
            _, topi = decoder_output.topk(1) # [64, 1]

            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            loss += F.cross_entropy(decoder_output, target_variable[t], ignore_index=EOS_token)

    loss.backward()

    clip = 50.0
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / max_target_len 


def trainIters(corpus, reverse, n_iteration, learning_rate, batch_size, n_layers, hidden_size,
                print_every, save_every, dropout, loadFilename=None, attn_model='dot', decoder_learning_ratio=5.0):

    voc, pairs = loadPrepareData(corpus)
    embedding_dict = concate_embedding(pairs, voc, hidden_size)

    # training data
    corpus_name = os.path.split(corpus)[-1].split('.')[0]
    training_batches = None
    try:
        training_batches = torch.load(os.path.join(save_dir, 'training_data', corpus_name,
                                                   '{}_{}_{}.tar'.format(n_iteration, \
                                                                         filename(reverse, 'training_batches'), \
                                                                         batch_size)))
    except FileNotFoundError:
        print('Generating training batches...')
        training_batches = [batch2TrainData([random.choice(pairs) for _ in range(batch_size)], voc, reverse)
                              for _ in range(n_iteration)]
        torch.save(training_batches, os.path.join(save_dir, 'training_data', corpus_name,
                                                  '{}_{}_{}.tar'.format(n_iteration, \
                                                                            filename(reverse, 'training_batches'), \
                                                                            batch_size)))

    # model
    checkpoint = None
    print('Building encoder and decoder ...')
    encoder = EncoderRNN(hidden_size, batch_size, n_layers, dropout)
    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, hidden_size, batch_size, voc.loc_count, n_layers, dropout)
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
    # use cuda
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # optimizer
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])

    # initialize
    print('Initializing ...')
    start_iteration = 1
    perplexity = []
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1
        perplexity = checkpoint['plt']

    for iteration in tqdm(range(start_iteration, n_iteration + 1)):
        training_batch = training_batches[iteration - 1]
        input_vec, input_lengths, target_vec, max_target_len = training_batch
        # print("input_lengths:", input_lengths)

        loss = train(input_vec, input_lengths, target_vec, max_target_len, encoder,
                     decoder, embedding_dict, encoder_optimizer, decoder_optimizer, batch_size)
        print_loss += loss
        perplexity.append(loss)

        if iteration % print_every == 0:
            print_loss_avg = math.exp(print_loss / print_every)
            print('%d %d%% %.4f' % (iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, 'model', corpus_name, '{}-{}_{}'.format(n_layers, batch_size, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'plt': perplexity
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, filename(reverse, 'backup_bidir_model'))))

if __name__ == "__main__":
    corpus = "data/greeting.txt"
    voc, pairs = loadPrepareData(corpus)
    corpus_name = os.path.split(corpus)[-1].split('.')[0]
    # print(corpus_name)

    hidden_size = 768

    embedding_dict = concate_embedding(pairs, voc, hidden_size)
    print(len(embedding_dict))
    print(embedding_dict[3])
    print(embedding_dict[0])

    print('Generating training batches...')
    n_iteration = 10
    batch_size = 16
    reverse = False

    training_batches = [batch2TrainData([random.choice(pairs) for _ in range(batch_size)], voc, reverse)
                        for _ in range(n_iteration)]
    input_variable, lengths, target_variable, max_target_len = training_batches[0]
    print(lengths)
    # print(input_variable)
    # print(input_variable.size())

    # start_tensor = get_sos(target_variable)
    # print(start_tensor)
    # print(start_tensor.shape)
    #
    # hidden_size = 768
    # n_layers = 1
    # dropout = 0
    #
    # print("Encoder ")
    # encoder = EncoderRNN(hidden_size, n_layers, dropout)
    # encoder = encoder.to(device)
    #
    # input_variable = input_variable.permute(1, 0, 2)
    # input_variable = input_variable.to(device)
    #
    # encoder_outputs, encoder_hidden = encoder(input_variable, lengths, None)
    # print(encoder_outputs.shape)
