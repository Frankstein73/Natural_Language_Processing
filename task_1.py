import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = '../python_data/data/'
data = pd.read_csv(PATH + 'train.tsv', sep='\t')


class Ngram:
    def __init__(self, data, ngram=None, iters=1000, lower_case=True):
        self.phrase_list = data['Phrase'][:iters]
        self.sentiment = data['Sentiment'][:iters].to_numpy()
        self.len = iters
        self.word_bags = {}
        self.lower_case = lower_case
        self.ngram = ngram if ngram else [1]  # while ngram=None it's a bag of words
        self.features = None
        if self.lower_case:
            self.phrase_list = [phrase.lower() for phrase in self.phrase_list]
        for phrase in self.phrase_list:
            for gram in self.ngram:
                words = phrase.split()
                for i in range(len(words) - gram + 1):
                    word = ' '.join(words[i:i + gram])
                    if word not in self.word_bags:
                        self.word_bags[word] = len(self.word_bags)

        self.features = np.zeros(shape=(len(self.phrase_list), len(self.word_bags) + 1))
        for i, phrase in enumerate(self.phrase_list):
            for gram in self.ngram:
                words = phrase.split()
                for j in range(len(words) - gram + 1):
                    word = ' '.join(words[j:j + gram])
                    # if word not in self.word_bags:
                    #     print("!!!")
                    self.features[j][self.word_bags[word]] += 1
            self.features[i][len(self.word_bags)] = 1

    def data_split(self, per=0.7):
        train_len = int(self.len * per)
        return self.features[:train_len], self.sentiment[:train_len], self.features[train_len:], self.sentiment[
                                                                                                 train_len:]


class SoftmaxRegression:
    def __init__(self, num_type):
        self.num_features = None
        self.num_sample = None
        self.num_type = num_type
        self.W = None

    def softmax(self, vector_x):
        vector_x = np.exp(vector_x - np.max(vector_x))
        return vector_x / np.sum(vector_x)

    def one_hot(self, y):
        vector_y = np.zeros((self.num_type, 1))
        vector_y[y] = 1
        return vector_y

    def calculate(self, X):
        WX = X.dot(self.W)
        WX -= np.max(WX, axis=1, keepdims=True)
        WX = np.exp(WX)
        WX /= np.sum(WX, axis=1, keepdims=True)
        return np.argmax(WX, axis=1)

    def predict(self, X, y):
        num_sample, num_features = X.shape
        print(num_sample, num_features)
        # correct = sum([y[i] == self.calculate(X[i]) for i in range(num_sample)]) / num_sample
        y_pred = self.calculate(X)
        correct = 0
        for i in range(num_sample):
            if y[i] == y_pred[i]:
                correct += 1
        print(correct / num_sample)

    def cross_entropy(self, y, y_pred):
        return -np.multiply(y, np.log(y_pred))

    def fit(self, X, y, lr=1, epochs=7000, strategy='mini_size', mini_size=100):
        self.num_sample, self.num_features = X.shape
        self.W = np.ones((self.num_features, self.num_type))

        if strategy == 'mini_size':
            for epoch in range(epochs):
                grad = np.zeros((self.num_features, self.num_type))
                for i in range(mini_size):
                    id = random.randint(0, self.num_sample - 1)
                    y_pred = self.softmax(self.W.T.dot(X[id].reshape(-1, 1)))
                    grad += X[id].reshape(-1, 1).dot((self.one_hot(y[id]) - y_pred).T)
                    # print(self.cross_entropy(self.one_hot(y[id]), y_pred)[y[id]])
                self.W += lr * grad / mini_size

        if strategy == 'shuffle':
            for epoch in range(epochs * mini_size):
                grad = np.zeros((self.num_features, self.num_type))
                id = random.randint(0, self.num_sample - 1)
                y_pred = self.softmax(self.W.T.dot(X[id].reshape(-1, 1)))
                grad += X[id].reshape(-1, 1).dot((self.one_hot(y[id]) - y_pred).T)
                self.W += lr * grad

        if strategy == 'batch':
            for epoch in range(epochs):
                grad = np.zeros((self.num_features, self.num_type))
                for id in range(self.num_sample):
                    y_pred = self.softmax(self.W.T.dot(X[id].reshape(-1, 1)))
                    grad += X[id].reshape(-1, 1).dot((self.one_hot(y[id]) - y_pred).T)

                self.W += lr * grad / self.num_sample


bag = Ngram(data=data, ngram=[1, 2, 3])
X_train, y_train, X_valid, y_valid = bag.data_split()
reg = SoftmaxRegression(num_type=5)
reg.fit(X_train, y_train, strategy='batch')

reg.predict(X_train, y_train)
reg.predict(X_valid, y_valid)
