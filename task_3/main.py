import pandas as pd
import torch
import tqdm
from torch import nn
from torchtext import vocab

from ESIM import ESIM
from data_processing import data_process

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH = '../../python_data/data/snli_1.0/snli_1.0'
total = 100000
per = 0.99
batch_size = 32
hidden_size = 128
num_type = 3
dropout = 0.3
epochs = 100

data = pd.read_json(PATH + '/snli_1.0_train.jsonl', lines=True, typ='series')
glove = vocab.GloVe(name='6B', dim=300)
features, labels = data_process(glove, data[:total])
len_train = int(total * per)
X_train, y_train, X_valid, y_valid = features[:len_train].to(device), labels[:len_train].to(device), features[
                                                                                                     len_train:].to(
    device), labels[len_train:].to(device)
net = ESIM(glove.vectors, X_train.shape[-1], hidden_size, num_type, dropout).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=4e-4)
loss_history = []
for epoch in range(epochs):
    pb = tqdm.tqdm(total=len_train // batch_size)
    for i in range(0, len_train, batch_size):
        optimizer.zero_grad()
        input = X_train[i:i + batch_size]
        label = y_train[i:i + batch_size]
        output = net(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        with torch.no_grad():
            pred = torch.argmax(net(X_valid), dim=-1)
            err = pred - y_valid
            pb.set_description(f'Epoch [{epoch}/{epochs}]')
            pb.set_postfix(loss=sum(loss_history[-10:]) / len(loss_history[-10:]),
                           valid_error=torch.count_nonzero(err) / len(err))
            pb.update()
    pb.close()
    torch.save(net.state_dict(), "result.pth")
