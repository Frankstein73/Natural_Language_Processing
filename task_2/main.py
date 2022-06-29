import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import data_processing as dp
from network import TextCNN

device = "cuda" if torch.cuda.is_available() else "cpu"
PATH = '../../python_data/data/'
NUM_TYPE = 5
num_sample = 100000
per = 0.7
length = 50
epochs = 20
lr = 1
embedding_size = 10
features_size = 2
kernel = [2, 3, 4]
dropout = 0.5
batch_size = 20

if __name__ == '__main__':
    data = pd.read_csv(PATH + 'train.tsv', sep='\t')
    X_train, y_train, X_valid, y_valid, num_embedding = dp.data_processing(data, per, num_sample, length, NUM_TYPE)

    net = TextCNN(num_embeddings=num_embedding, length=length, embedding_size=embedding_size,
                  feature_size=features_size, kernel_size=kernel, dropout=dropout, num_type=NUM_TYPE).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
    loss_history = []
    # net = torch.load('net.pkl')
    # print("!")
    for epoch in range(epochs):
        print(epoch)
        for st in range(0, int(num_sample * per) - batch_size, batch_size):
            optimizer.zero_grad()
            input = torch.from_numpy(X_train[st:st + batch_size]).to(device)
            output = torch.from_numpy(y_train[st:st + batch_size]).to(device)
            pred = net(input)
            loss = criterion(pred, output)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
    # print(loss_history[-1])
    corr = 0

    torch.save(net, 'net.pkl')
    with torch.no_grad():

        for st in range(0, int(num_sample * per) - batch_size, batch_size):
            input = torch.from_numpy(X_train[st:st + batch_size]).to(device)
            output = torch.from_numpy(y_train[st:st + batch_size]).to(device)
            pred = net(input)
            output = torch.argmax(output, dim=1)
            pred = torch.argmax(pred, dim=1)
            corr += torch.sum(pred == output).item()

        print(corr)
        corr = 0
        for st in range(0, int(num_sample * (1 - per)) - batch_size, batch_size):
            input = torch.from_numpy(X_valid[st:st + batch_size]).to(device)
            output = torch.from_numpy(y_valid[st:st + batch_size]).to(device)
            pred = net(input)
            output = torch.argmax(output, dim=1)
            pred = torch.argmax(pred, dim=1)
            corr += torch.sum(pred == output).item()
            # print(output)
            # print(pred)
        print(corr)
