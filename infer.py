import numpy as np
import sys
from argparse import ArgumentParser
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchtext import data
from torchtext import datasets
from data import StanfordNLI, StanfordNLI_test

parser = ArgumentParser()
parser.add_argument('--model_path',
                    type=str,
                    default='results/final_LSTMEncoder_2048D.pt')
parser.add_argument('--batch_size',
                    type=int,
                    default=64)
parser.add_argument('--custom',
                    type=bool,
                    default=False)
parser.add_argument('--hypothesis',
                    type=str,
                    default='a beautiful bride walking on a sidewalk with her new husband .')
parser.add_argument('--premise',
                    type=str,
                    default='a beautiful bride walking on a sidewalk with her new husband .')

class example(object):
    premise = torch.FloatTensor()
    hypothesis = torch.FloatTensor()

def user_input():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    config = parser.parse_args()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)

    inputs = data.Field(lower=config.lower, tokenize='spacy')
    labels = data.Field(sequential=False, unk_token=None)
    category_field = data.Field(sequential=False)
    id_field = data.Field(sequential=False, unk_token=None)

    test = StanfordNLI_test.splits(inputs, labels)[0]

    if torch.cuda.is_available() != True:
        _, test = test.split(0.9)
    
    inputs.build_vocab(test)
    labels.build_vocab(test)

    test_iter = data.BucketIterator(test,
        batch_size=config.batch_size,
        device=device)

    hypo_emb = []
    prem_emb = []
    example_input = example()
    for element in config.hypothesis.split():
        hypo_emb.append(inputs.vocab.stoi[element])
    for element in config.premise.split():
        prem_emb.append(inputs.vocab.stoi[element])

    example_input.premise = torch.LongTensor(prem_emb).to(device=device)
    example_input.premise = example_input.premise.view(-1,1)
    example_input.hypothesis = torch.LongTensor(hypo_emb).to(device=device)
    example_input.hypothesis = example_input.hypothesis.view(-1,1)

    # Loss
    criterion = nn.CrossEntropyLoss()
    torch.nn.Module.dump_patches = True
    test_model = torch.load(config.model_path)

    # Switch model to evaluation mode
    test_model.eval()
    test_iter.init_epoch()
    print(config.premise)
    print(config.hypothesis)
    print('PREDICTION')
    answer = test_model(example_input)
    for j, label in enumerate(answer[0]):
        if label == torch.max(answer[0]).item():
            print(labels.vocab.itos[j], end=' ')
def main():

    config = parser.parse_args()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)

    inputs = data.Field(lower=config.lower, tokenize='spacy')
    labels = data.Field(sequential=False, unk_token=None)
    category_field = data.Field(sequential=False)
    id_field = data.Field(sequential=False, unk_token=None)

    test = StanfordNLI_test.splits(inputs, labels)[0]
    if torch.cuda.is_available() != True:
        _, test = test.split(0.9)

    inputs.build_vocab(test)
    labels.build_vocab(test)

    test_iter = data.BucketIterator(test,
        batch_size=config.batch_size,
        device=device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    test_model = torch.load(config.model_path)

    # Switch model to evaluation mode
    test_model.eval()
    test_iter.init_epoch()

    n_test_correct = 0
    test_loss = 0
    test_losses = []
    for test_batch_id, test_batch in enumerate(test_iter):
        answer = test_model(test_batch)
        uid = 1+test_batch_id*config.batch_size
        for i in range(test_batch.batch_size):
            for prem in test_batch.premise.transpose(0,1)[i]:
                x = prem.item()
                if not inputs.vocab.itos[x] == '<pad>':
                    print(inputs.vocab.itos[x], end=' ')
            print('|', end=' ')
            for hypo in test_batch.hypothesis.transpose(0,1)[i]:
                y = hypo.item()
                if not inputs.vocab.itos[y] == '<pad>':
                    print(inputs.vocab.itos[y], end=' ')
            print('|', end=' ')
            for j, label in enumerate(answer[i]):
                if label.item() == torch.max(answer[i]).item():
                    print(labels.vocab.itos[j], end=' ')
                    if j == test_batch.label[i].item():
                        print('| CORRECT |', end=' ')
                        print(labels.vocab.itos[test_batch.label[i].item()], end=' ')
                    else:
                        print('| INCORRECT |', end=' ')
                        print(labels.vocab.itos[test_batch.label[i].item()], end=' ')

        n_test_correct += (torch.max(answer, 1)[1].view(test_batch.label.size()).data == \
            test_batch.label.data).sum()
        test_loss = criterion(answer, test_batch.label)
        test_losses.append(test_loss.item())

    test_acc = 100. * n_test_correct / len(test)

    print('\nLoss: {:.4f} / Accuracy: {:.4f}\n'.format(round(np.mean(test_losses), 2), test_acc))

if __name__ == '__main__':
    if config.custom == True:
        user_input()
    else:
        main()