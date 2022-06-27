import pandas as pd
import random
import numpy as np
import itertools


def data_processing(data, per, num_sample, length, num_type):
    start_point = random.randint(0, len(data) - num_sample - 1)
    start_point = 0
    phraseList = data['Phrase'][start_point:num_sample + start_point]
    sentiment = data['Sentiment'][start_point:num_sample + start_point].to_numpy()
    phrase_list = [phrase.lower() for phrase in phraseList]
    id = 0
    X = np.zeros([num_sample, length])
    y = np.zeros([num_sample, num_type])
    wordsbag = {}
    for i in range(num_sample):
        y[i][sentiment[i]] = 1
    for phrase in phrase_list:
        words = phrase.split()
        for word in words:
            if word not in wordsbag:
                wordsbag[word] = len(wordsbag) + 1

    for phrase in phrase_list:
        words = phrase.split()
        for j in range(length):
            if j < len(words):
                X[id][j] = wordsbag[words[j]]
        id += 1
    X = X.astype(int)
    train_len = int(num_sample * per)
    num_embed = len(wordsbag) + 1
    return X[:train_len], y[:train_len], X[train_len:], y[train_len:], num_embed


if __name__ == '__main__':
    PATH = '../../python_data/data/'
    data = pd.read_csv(PATH + 'train.tsv', sep='\t')
    X_train, y_train, X_valid, y_valid, n = data_processing(data, 0.7, 10, 16, 5)
    print(X_train)
