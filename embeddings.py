import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class SentenceEmbedding(nn.Module):
    def __init__(self, config):
        super(SentenceEmbedding, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config.embed_size, config.embed_dim)
        self.encoder = eval(config.encoder_type)(config)

    def forward(self, input_sentence):
        sentence = self.word_embedding(input_sentence)
        embedding = self.encoder(sentence)
        return embedding

    def encode(self, input_sentence):
        embedding = self.encoder(sentence)
        return embedding


class MeanEmbedding(nn.Module):
    def __init__(self, config):
        super(MeanEmbedding, self).__init__()
        self.config = config

    def forward(self, inputs):
        embedding = inputs.mean(0)
        return embedding

class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(input_size=config.embed_dim,
                           hidden_size=config.hidden_dim,
                           num_layers=config.layers,
                           dropout=config.dropout,
                           bidirectional=False)
        self.batch_norm = nn.BatchNorm1d(config.hidden_dim)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        embedding = self.rnn(inputs)
        embedding = embedding[0].mean(0)
        #embedding = embedding.squeeze(0)
        #embedding = self.batch_norm(embedding)
        return embedding

class BiLSTMMaxPoolEncoder(nn.Module):
    def __init__(self, config):
        super(BiLSTMMaxPoolEncoder, self).__init__()
        self.config = config
        self.rnn1 = nn.LSTM(input_size=config.embed_dim,
                           hidden_size=config.hidden_dim,
                           num_layers=config.layers,
                           dropout=config.dropout,
                           bidirectional=True)

        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        embedding = self.rnn1(inputs)[0]
        emb = self.max_pool(embedding.permute(1,2,0))
        emb = emb.squeeze(2)
        return emb

class BiLSTMEncoder(nn.Module):
    def __init__(self, config):
        super(BiLSTMEncoder, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(input_size=config.embed_dim,
                           hidden_size=config.hidden_dim,
                           num_layers=config.layers,
                           dropout=config.dropout,
                           bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(2*config.hidden_dim)
    def forward(self, inputs):
        batch_size = inputs.size()[1]
        embedding = self.rnn(inputs)
        embedding = embedding[0].mean(0)
        #embedding = embedding.squeeze(0)
        embedding = self.batch_norm(embedding)
        return embedding