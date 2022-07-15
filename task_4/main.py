import torch
import tqdm
from network import BiLstmCRF
from opti import *
from utils import *


def main():
    total_loss = 0
    mod = 100
    f1 = 0
    f0 = 0
    loss_history = []
    train_dataloader, valid_dataloader, max_length, tag_to_index, vocab_size = build_dataloader(batch_size_train,
                                                                                                batch_size_valid)
    num_tags = len(tag_to_index)
    net = BiLstmCRF(vocab_size=vocab_size, embedding_dim=ebd_dim, tag_to_index=tag_to_index, hidden_size=hidden_size,
                    max_length=max_length).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    for epoch in range(epochs):

        for step, (inputs, tags, length_list) in enumerate(train_dataloader):
            net.zero_grad()
            inputs = inputs.to(device)
            tags = tags.to(device)
            length_list = length_list.to(device)
            loss = -net(inputs, tags, length_list)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            _, f0_value = get_f1_value(net.predict(inputs, length_list), tags, length_list, num_tags)
            f0 += f0_value
            if step and step % mod == 0:
                print('f0:', f0 / mod)
                f0 = 0
                test(net, valid_dataloader, num_tags)


def test(net, valid_dataloader, num_tags):
    f1 = 0
    with torch.no_grad():
        for step, (inputs, tags, length_list) in enumerate(valid_dataloader):
            inputs = inputs.to(device)
            tags = tags.to(device)
            length_list = length_list.to(device)
            _, f1_value = get_f1_value(net.predict(inputs, length_list), tags, length_list, num_tags)
            f1 += f1_value
        print('f1:', f1 / len(valid_dataloader))
        f1 = 0


if __name__ == '__main__':
    main()
