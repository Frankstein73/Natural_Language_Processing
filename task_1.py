import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = '../python_data/data/'
train_data = pd.read_csv(PATH + 'train.tsv', sep='\t')
test_data = pd.read_csv(PATH + 'test.tsv', sep='\t')


class BagOfWolds:
    def __init__(self, phrase_list, lower_case=False):
        self.word_bags = {}
        self.lower_case = lower_case
        self.features = None
        if self.lower_case:
            phrase_list = [phrase.lower() for phrase in phrase_list]
        for phrase in phrase_list:
            words = phrase.split()
            for word in words:
                if word not in self.word_bags:
                    self.word_bags[word] = len(self.word_bags)

        self.features = np.zeros(shape=(len(phrase_list), len(self.word_bags) + 1))
        for i, phrase in enumerate(phrase_list):
            words = phrase.split()
            for word in words:
                self.features[i][self.word_bags[word]] += 1
            self.features[i][len(self.word_bags)] = 1


class Ngram:
    def __init__(self, phrase_list, ngram=None, lower_case=False):
        self.word_bags = {}
        self.lower_case = lower_case
        self.ngram = ngram if ngram else [1]
        self.features = None
        if self.lower_case:
            phrase_list = [phrase.lower() for phrase in phrase_list]
        for phrase in phrase_list:
            for gram in self.ngram:
                words = phrase.split()
                for i in range(len(words) - gram + 1):
                    word = ' '.join(words[i:i + gram])
                    if word not in self.word_bags:
                        self.word_bags[word] = len(self.word_bags)

        self.features = np.zeros(shape=(len(phrase_list), len(self.word_bags) + 1))
        for i, phrase in enumerate(phrase_list):
            for gram in self.ngram:
                words = phrase.split()
                for j in range(len(words) - gram + 1):
                    word = ' '.join(words[j:j + gram])
                    self.features[j][self.word_bags[word]] += 1
            self.features[i][len(self.word_bags)] = 1


def cross_entropy(y_tra, y_pre):
    return -np.multiply(y_tra, np.log(y_pre))


def softmax(x):
    x = np.exp(x - np.max(x))
    return x / np.sum(x)


class Regression:
    def __init__(self):
        # self.epochs = None
        # self.lr = None
        self.w = None
        self.loss = []
        self.accu = []

    def predict(self, X, y):
        num_phrases, _ = np.shape(X)
        pred = 0
        total = 0
        for i in range(num_phrases):
            total += 1
            y_pre = self.w @ X[i]
            pos = np.argmax(y_pre)
            # print(y_pre)
            # print(y[i])
            if y[i] == pos:
                pred += 1
            self.accu.append(pred / total)

    def fit(self, X, y, num_classes, epochs=1, lr=0.001, batch_size=1, type='shuffle',
            mini_batch=1000):
        self.loss = []
        num_phrases, num_features = np.shape(X)
        Y = np.zeros(shape=(num_phrases, num_classes))
        for i in range(num_phrases):
            Y[i][y[i]] = 1
        self.w = np.ones(shape=(num_classes, num_features))
        if type == 'batch':
            for epoch in range(epochs):
                grad = np.zeros_like(self.w)
                LOSS = 0
                for i in range(num_phrases):
                    y_pre = softmax(self.w @ X[i].T)
                    grad = np.subtract(grad, np.outer(X[i], Y[i] - y_pre).T)
                    loss = cross_entropy(y_tra=Y[i], y_pre=y_pre)[y[i]]
                    LOSS += loss
                self.loss.append(LOSS / num_phrases)
                grad /= num_phrases
                self.w -= lr * grad
        if type == 'shuffle':
            for epoch in range(epochs):
                id = random.randint(0, num_phrases - 1)
                grad = np.zeros_like(self.w)
                y_pre = softmax(self.w @ X[id].T)
                grad = np.subtract(grad, np.outer(X[id], Y[id] - y_pre).T)
                loss = cross_entropy(y_tra=Y[id], y_pre=y_pre)[y[id]]
                self.loss.append(loss)
                self.w -= lr * grad
                print(loss)

        if type == 'minibatch':
            for epoch in range(epochs):
                grad = np.zeros_like(self.w)
                LOSS = 0
                for i in range(mini_batch):
                    # print(i)
                    id = random.randint(0, num_phrases - 1)
                    y_pre = softmax(self.w @ X[id].T)
                    grad = np.subtract(grad, np.outer(X[id], Y[id] - y_pre).T)
                    loss = cross_entropy(y_tra=Y[id], y_pre=y_pre)[y[id]]
                    LOSS += loss
                self.loss.append(LOSS / mini_batch)
                grad /= mini_batch
                self.w -= lr * grad


X = train_data['Phrase']
y = train_data['Sentiment']
bag = BagOfWolds(phrase_list=X, lower_case=True)
X_features = bag.features
reg = Regression()
reg.fit(X=X_features, y=y, num_classes=5, epochs=10000, type='minibatch')
plt.plot(reg.loss)
plt.savefig('pic.png')
plt.show()
