from data import StanfordNLI
import torchtext.data as data
from argparse import ArgumentParser
import os
from embeddings import SentenceEmbedding
from classifier import NLIModel
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import glob
import torchtext.vocab as vocab
from tensorboardX import SummaryWriter

parser = ArgumentParser()
parser.add_argument('--epochs',
                    type=int,
                    default=20)
parser.add_argument('--batch_size',
                    type=int,
                    default=64)
parser.add_argument("--encoder_type",
                    type=str,
                    choices=['BiLSTMMaxPoolEncoder',
                             'LSTMEncoder',
                             'BiLSTMEncoder',
                             'MeanEmbedding'],
                    default='MeanEmbedding')
parser.add_argument("--activation",
                    type=str,
                    choices=['tanh', 'relu'],
                    default='relu')
parser.add_argument("--optimizer",
                    type=str,
                    choices=['adam',
                             'sgd'],
                    default='sgd')
parser.add_argument('--embed_dim',
                    type=int,
                    default=300)
parser.add_argument('--fc_dim',
                    type=int,
                    default=600)
parser.add_argument('--hidden_dim',
                    type=int,
                    default=600)
parser.add_argument('--layers',
                    type=int,
                    default=1)
parser.add_argument('--dropout',
                    type=float,
                    default=0.1)
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.1)
parser.add_argument('--lr_patience',
                    type=int,
                    default=1)
parser.add_argument('--lr_decay',
                    type=float,
                    default=0.99)
parser.add_argument('--lr_factor',
                    type=float,
                    default=0.2)
parser.add_argument('--weight_decay',
                    type=float,
                    default=0)
parser.add_argument('--word_embedding',
                    type=str,
                    default='glove.840B.300d')
parser.add_argument('--resume_snapshot',
                    type=str,
                    default='')
parser.add_argument('--early_stopping_patience',
                    type=int,
                    default=3)
parser.add_argument('--save_path',
                    type=str,
                    default='results')
parser.add_argument('--seed',
                    type=int,
                    default=1234)
parser.add_argument('--mini',
                    type=bool,
                    default=False)


if __name__ == '__main__':
    #writer = SummaryWriter()
    writer_val = SummaryWriter('runs/plot_val')
    writer_train = SummaryWriter('runs/plot_train')
    test_time = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    config = parser.parse_args()
    config.device = device
    inputs = data.Field(lower=config.lower, tokenize='spacy')
    labels = data.Field(sequential=False, unk_token=None, tokenize='spacy')
    train, dev, test = StanfordNLI.splits(inputs, labels)

    glove_embeddings = vocab.Vectors('glove.840B.300d.txt', '.vector_cache/') 

    max_vocab =int(len(glove_embeddings.itos))

    if torch.cuda.is_available() != True or config.mini:
        _, train = train.split(0.9)
        _, dev = dev.split(0.9)
        _, test = test.split(0.9)

    inputs.build_vocab(train, 
                     max_size = max_vocab, 
                     vectors = glove_embeddings)
    
    #converts text labels into numeric data
    labels.build_vocab(train)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                                                                 batch_size=config.batch_size,
                                                                 device=device)
    config.embed_size = len(inputs.vocab)+1
    config.out_dim = len(labels.vocab)
    config.cells = config.layers
    model = NLIModel(config)

    if config.resume_snapshot:
        if torch.cuda.is_available():
            model = torch.load(config.resume_snapshot,
                           map_location=lambda storage, location: storage.cuda(device))
        else:
            model = torch.load(config.resume_snapshot)
    else:
        model = NLIModel(config)
        if config.word_embedding:
            model.sentence_embedding.word_embedding.weight.data = inputs.vocab.vectors
            if torch.cuda.is_available():
                model.to(device=device)

    criterion = nn.CrossEntropyLoss().to(device=device)

    if config.optimizer == 'sgd':
        optim_algorithm = optim.SGD
    elif config.optimizer == 'adam':
        optim_algorithm = optim.Adam

    optimizer = optim_algorithm(model.parameters(),
                                lr=config.learning_rate,
                                weight_decay=config.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               'min',
                                               factor=config.lr_factor,
                                               patience=config.lr_patience,
                                               verbose=False,
                                               min_lr=1e-5)
    iterations = 0
    best_dev_acc = -1
    dev_accuracies = []
    best_dev_loss = 1
    early_stopping = 0
    stop_training = False
    train_iter.repeat = False
    params = sum([p.numel() for p in model.parameters()])
    print('Model:\n')
    print(model)
    print('\n')
    print('Parameters: {}'.format(params))
    for epoch in range(config.epochs):
        if stop_training == True:
            break

        train_iter.init_epoch()
        n_correct = 0
        n_total = 0
        all_losses = []
        train_accuracies = []
        all_losses = []

        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * config.lr_decay if epoch>0 and config.optimizer == 'sgd' else optimizer.param_groups[0]['lr']
        print('\nEpoch: {:>02.0f}/{:<02.0f}'.format(epoch+1, config.epochs), end=' ')
        print('(Learning rate: {})'.format(optimizer.param_groups[0]['lr']))

        for batch_id, batch in enumerate(train_iter):
            model.train()
            optimizer.zero_grad()
            iterations += 1

            answer = model(batch)
            n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
            n_total += batch.batch_size
            train_acc = 100. * n_correct/n_total
            train_accuracies.append(train_acc.item())

            loss = criterion(answer, batch.label)
            all_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            print('Progress: {:3.0f}% - Batch: {:>4.0f}/{:<4.0f} - Loss: {:6.2f}% - Accuracy: {:6.2f}%'.format(
            100. * (1+batch_id) / len(train_iter),
            1+batch_id, len(train_iter),
            round(100. * np.mean(all_losses), 2),
            round(np.mean(train_accuracies), 2)), end='\r')
            writer_train.add_scalar('Train/Loss/' + config.encoder_type, round(100. * np.mean(all_losses), 2), iterations)
            writer_train.add_scalar('Train/Acc/' + config.encoder_type, round(np.mean(train_accuracies), 2), iterations)


            if 1+batch_id == len(train_iter):
                model.eval()
                dev_iter.init_epoch()

                # Calculate Accuracy
                n_dev_correct = 0
                dev_loss = 0
                dev_losses = []
                dev_losses2 = 0

                for dev_batch_id, dev_batch in enumerate(dev_iter):
                    answer = model(dev_batch)
                    n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == \
                        dev_batch.label.data).sum()
                    dev_loss = criterion(answer, dev_batch.label)
                    dev_losses.append(dev_loss.item())
                    dev_losses2 += dev_loss.item()


                dev_acc = 100. * n_dev_correct / len(dev)
                dev_loss2 = 100. * dev_losses2 / len(dev)
                dev_accuracies.append(dev_acc.item())

                print('\nDev loss: {}% - Dev accuracy: {}%'.format(round(100.*np.mean(dev_losses), 4), dev_acc))

                writer_val.add_scalar('Val/Loss/' + config.encoder_type, round(100.*np.mean(dev_losses), 4), (epoch+1))
                writer_val.add_scalar('Val/Acc/' + config.encoder_type, dev_acc, (epoch+1))

                if dev_acc > best_dev_acc:

                    best_dev_acc = dev_acc
                    best_dev_epoch = 1+epoch
                    snapshot_prefix = os.path.join(config.save_path, 'best')
                    dev_snapshot_path = snapshot_prefix + \
                        '_{}_{}D_devacc_{}_epoch_{}.pt'.format(config.encoder_type, config.hidden_dim, dev_acc, 1+epoch)

                    # save model, delete previous snapshot
                    torch.save(model, dev_snapshot_path)
                    for f in glob.glob(snapshot_prefix + '*'):
                        if f != dev_snapshot_path:
                            os.remove(f)

                # Check for early stopping
                if np.mean(dev_losses) < best_dev_loss:
                    best_dev_loss = np.mean(dev_losses)
                else:
                    early_stopping += 1

                if early_stopping > config.early_stopping_patience and config.optimizer != 'sgd':
                    stop_training = True
                    print('\nEarly stopping')

                if config.optimizer == 'sgd' and optimizer.param_groups[0]['lr'] < 1e-5:
                    stop_training = True
                    print('\nEarly stopping')

                # Update learning rate
                scheduler.step(round(np.mean(dev_losses), 2))
                dev_losses = []


            # If training has completed, calculate the test scores
            if stop_training == True or (1+epoch == config.epochs and 1+batch_id == len(train_iter)):
                print('\nTraining completed after {} epocs.\n'.format(1+epoch))

                last_model_prefix = os.path.join(config.save_path, 'final')
                last_model_path = last_model_prefix + \
                '_{}_{}D.pt'.format(config.encoder_type, config.hidden_dim)
                torch.save(model, last_model_path)
                for f in glob.glob(last_model_prefix + '*'):
                    if f != last_model_path:
                        os.remove(f)

                test_model = torch.load(dev_snapshot_path)
                test_model.eval()
                test_iter.init_epoch()

                n_test_correct = 0
                test_loss = 0
                test_losses = []

                for test_batch_id, test_batch in enumerate(test_iter):
                    answer = test_model(test_batch)
                    n_test_correct += (torch.max(answer, 1)[1].view(test_batch.label.size()).data == \
                        test_batch.label.data).sum()
                    test_loss = criterion(answer, test_batch.label)
                    test_losses.append(test_loss.item())

                test_acc = 100. * n_test_correct / len(test)
    writer.close()