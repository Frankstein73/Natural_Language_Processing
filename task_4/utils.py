import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from opti import *


class MyDataset(Dataset):
    def __init__(self, x, y, length_list):
        self.x = x
        self.y = y
        self.length_list = length_list

    def __getitem__(self, item):
        data = self.x[item]
        labels = self.y[item]
        length = self.length_list[item]
        return data, labels, length

    def __len__(self):
        return len(self.x)


def read_file(path, length):
    sentence_list = []
    sentence_label_list = []
    with open(path, 'r', encoding='UTF-8') as f:
        word_list = []
        word_label_list = []
        for line in f:
            line = line.strip()
            if not line:
                if word_list:
                    sentence_list.append(' '.join(word_list))
                    sentence_label_list.append(' '.join(word_label_list))
                    word_list = []
                    word_label_list = []
            else:
                line = line.split()
                assert len(line) == 4
                if line[0] == '-DOCSTART-':
                    continue
                word_list.append(line[0])
                word_label_list.append(line[3])

        if word_list:
            sentence_list.append(word_list)
            sentence_label_list.append(word_label_list)
    return sentence_list[:length], sentence_label_list[:length]


def get_word_index(word, index):
    if index[word] is not None:
        return index[word]
    else:
        return index['<unknown>']


def padding(sentence, max_length, index):
    length = len(sentence)
    for i in range(max_length - length):
        sentence.append(index['<pad>'])
    return sentence


def sentence_to_index(sentence, index):
    return [get_word_index(word, index) for word in sentence.split()]


def build_vocab(sentence_list):
    res = []
    temp = {}
    for sentence in sentence_list:
        res += [word for word in sentence.split()]
    for word in res:
        if word not in temp:
            temp[word] = len(temp)
    return list(temp.keys())


def get_data():
    X_train, y_train = read_file(PATH + '/eng.train', train_length)
    X_valid, y_valid = read_file(PATH + '/eng.testa', valid_length)
    return X_train, y_train, X_valid, y_valid


def data_processing(X_train, y_train, X_valid, y_valid):
    # X_train, y_train, X_valid, y_valid = get_data()
    index_word = build_vocab(X_train + X_valid)
    index_tag = build_vocab(y_train + y_valid)
    word_to_index = {index_word[i]: i for i in range(len(index_word))}
    tag_to_index = {index_tag[i]: i for i in range(len(index_tag))}
    word_to_index['<pad>'] = len(word_to_index)
    word_to_index['<unknown>'] = len(word_to_index)
    tag_to_index['<START>'] = len(tag_to_index)
    tag_to_index['<STOP>'] = len(tag_to_index)
    tag_to_index['<pad>'] = len(tag_to_index)
    vocab_size = len(word_to_index)
    # index_to_tag = {value: key for key, value in tag_to_index.items()}
    return word_to_index, tag_to_index, vocab_size


def build_dataloader(batch_size_train, batch_size_valid):
    X_train, y_train, X_valid, y_valid = get_data()
    word_to_index, tag_to_index, vocab_size = data_processing(X_train, y_train, X_valid, y_valid)
    input_train = [sentence_to_index(sentence, word_to_index) for sentence in X_train]
    input_valid = [sentence_to_index(sentence, word_to_index) for sentence in X_valid]

    tags_train = [sentence_to_index(tag, tag_to_index) for tag in y_train]
    tags_valid = [sentence_to_index(tag, tag_to_index) for tag in y_valid]

    length_list_train = [len(sentence) for sentence in input_train]
    length_list_valid = [len(sentence) for sentence in input_valid]
    max_length = max(max(length_list_train), max(length_list_valid))
    input_train = torch.tensor([padding(sentence, max_length, word_to_index) for sentence in input_train])
    tags_train = torch.tensor([padding(tag, max_length, tag_to_index) for tag in tags_train])
    input_valid = torch.tensor([padding(sentence, max_length, word_to_index) for sentence in input_valid])
    tags_valid = torch.tensor([padding(tag, max_length, tag_to_index) for tag in tags_valid])

    train_dataset = MyDataset(input_train, tags_train, length_list_train)
    valid_dataset = MyDataset(input_valid, tags_valid, length_list_valid)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size_train)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size_valid)
    return train_dataloader, valid_dataloader, max_length, tag_to_index, vocab_size


def get_f1_value(preds, tags, length_list, num_tags):
    tag_table = torch.zeros((num_tags, num_tags))
    f1_tags = []

    for i, length in enumerate(length_list):
        for j in range(length):
            tag_table[tags[i][j]][preds[i][j]] += 1
    for tag in range(num_tags):
        num_pre = torch.sum(tag_table[:, tag]).item()
        num_act = torch.sum(tag_table[tag]).item()
        num_corr = tag_table[tag][tag].item()
        try:
            f1_tags.append(2 / (num_pre / num_corr + num_act / num_corr))
        except ZeroDivisionError:
            f1_tags.append(0)

    return f1_tags, sum(f1_tags) / np.count_nonzero(np.array(f1_tags))
