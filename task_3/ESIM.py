import torch
import torch.nn as nn
import torch.nn.functional as F


class ESIM(nn.Module):
    def __init__(self, word_dict, word_num, hidden_size, num_type, dropout):
        super(ESIM, self).__init__()
        self.word_num = word_num
        self.vector_size = word_dict.shape[1]
        self.embedding = nn.Embedding.from_pretrained(word_dict)
        self.input_encoder = nn.LSTM(input_size=self.vector_size, hidden_size=hidden_size, batch_first=True,
                                     bidirectional=True)
        self.composition = nn.LSTM(input_size=hidden_size * 8, hidden_size=hidden_size, batch_first=True,
                                   bidirectional=True)

        # self.avg_pool = F.avg_pool2d(kernel_size=(self.word_num, 1))
        # self.max_pool = F.max_pool2d(kernel_size=(self.word_num, 1))
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size * 8),
            nn.Linear(hidden_size * 8, num_type),
            nn.Tanh(),
            nn.BatchNorm1d(num_type),
            nn.Linear(num_type, num_type),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = x.permute(1, 0, 2)  # [2, batch_size, word_num]
        embed_x_1 = self.embedding(x[0])  # [batch_size, word_num, vector_size]
        embed_x_2 = self.embedding(x[1])
        a_bar, _ = self.input_encoder(embed_x_1)  # [batch_size, word_num, hidden_size * num_directions]
        b_bar, _ = self.input_encoder(embed_x_2)

        E = torch.matmul(a_bar, torch.transpose(b_bar, 1, 2))  # [batch_size, word_num, word_num]

        a_tilde = torch.matmul(torch.softmax(E, dim=2), b_bar)  # [batch_size,word_num,hidden_size * num_directions]
        b_tilde = torch.matmul(torch.transpose(torch.softmax(E, dim=1), 1, 2), a_bar)
        m_a = torch.cat([a_bar, a_tilde, a_bar - a_tilde, a_bar * a_tilde],
                        dim=2)  # [batch_size,word_num,hidden_size * num_directions * 4]
        m_b = torch.cat([b_bar, b_tilde, b_bar - b_tilde, b_bar * b_tilde],
                        dim=2)  # [batch_size,word_num,hidden_size * num_directions * 4]
        v_a, _ = self.composition(m_a)
        v_b, _ = self.composition(m_b)  # [batch_size, word_num, hidden_size * num_directions]
        # print(type(v_a))
        v = torch.cat([F.avg_pool2d(input=v_a, kernel_size=(self.word_num, 1)),
                       F.max_pool2d(input=v_a, kernel_size=(self.word_num, 1)),
                       F.avg_pool2d(input=v_b, kernel_size=(self.word_num, 1)),
                       F.max_pool2d(input=v_b, kernel_size=(self.word_num, 1))],
                      dim=2)  # [batch_size, 1, hidden_size * num_directions * 4]
        v = torch.squeeze(v, dim=1)  # [batch_size, hidden_size * num_directions * 4]
        res = self.mlp(v)
        return res
