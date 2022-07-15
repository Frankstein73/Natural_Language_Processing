import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_char, embedding_dim, filter_size, out_channel, dropout):
        super(CNN, self).__init__()
        self.num_char = num_char
        self.embedding_dim = embedding_dim
        self.filter_size = filter_size
        self.out_channel = out_channel
        self.dropout = dropout
        self.ebd = nn.Embedding(num_embeddings=num_char, embedding_dim=embedding_dim)
        self.cnn = nn.Conv2d(in_channels=1, out_channels=out_channel, kernel_size=(filter_size, embedding_dim))
        torch.nn.init.kaiming_uniform_(self.ebd.weight)

    def forward(self, x):  # [batch_size, num_word, char_size]
        batch_size, num_word, char_size = x.size()
        x = x.view(-1, char_size)  # [batch_size * num_word, char_size]
        x = self.ebd(x).unsqueeze(1)  # [batch_size * num_word, 1, char_size, embedding_dim]
        x = self.cnn(x)  # [batch_size * num_word, out_channel, char_size - filter_size + 1, 1]
        x = F.relu(x)
        x = F.max_pool2d(input=x, kernel_size=(x.size(2), 1)).squeeze()  # [batch_size * num_word, out_channel]
        x = F.dropout(x, p=self.dropout)
        x = x.view(batch_size, num_word, -1)  # [batch_size, num_word, out_channel]
        return x


class BiLSTM(nn.Module):
    def __init__(self, word_dict, num_word, hidden_size, num_type, dropout, num_char, char_embedding_dim, filter_size,
                 out_channel):
        super(BiLSTM, self).__init__()
        self.num_word = num_word
        self.hidden_size = hidden_size
        self.num_type = num_type
        self.dropout = dropout
        self.num_char = num_char
        self.char_embedding_dim = char_embedding_dim
        self.filter_size = filter_size
        self.out_channel = out_channel
        self.word_embedding_dim = word_dict.shape[1]
        self.ebd = nn.Embedding.from_pretrained(word_dict)
        self.cnn = CNN(num_char, char_embedding_dim, filter_size, out_channel, dropout)
        self.lstm = nn.LSTM(input_size=out_channel + self.word_embedding_dim, hidden_size=hidden_size, batch_first=True,
                            bidirectional=True)
        self.logistic = nn.Sequential(
            nn.Linear(hidden_size * 2, num_type),
            nn.ReLU()
        )

    def forward(self, x_word, x_char):
        x_word = self.ebd(x_word)  # [batch_size, num_word, word_embedding_dim]
        x_char = self.cnn(x_char)  # [batch_size, num_word, out_channel]
        x = torch.cat([x_word, x_char], dim=2)  # [batch_size, num_word, word_embedding_dim + out_channel]
        res, _ = self.lstm(x)  # [batch_size, num_word, num_direction * hidden_size]
        res = self.logistic(res)  # [batch_size, num_word, num_type]
        return res


def sum_torch_list(torch_list):
    res = torch.zeros_like(torch_list[0])
    for i in range(len(torch_list)):
        res += torch_list[i]
    return res


class CRF(nn.Module):
    def __init__(self, num_type):
        super(CRF, self).__init__()
        self.num_type = num_type
        self.translation = nn.Parameter(torch.randn(num_type + 2, num_type + 2))
        nn.init.xavier_uniform_(self.transitions)
        self.translation.data[:, 0] = -1000
        self.translation.data[num_type + 1, :] = -1000

    def _score_path(self, path_tag, word_tag_value):
        # path_tag : [batch_size, word_size]
        # word_tag_value : [batch_size, word_size, num_type]
        batch_size, word_size = path_tag.size()
        score_1 = torch.zeros((batch_size, 1))
        score_2 = torch.zeros((batch_size, 1))
        score = torch.zeros((batch_size, 1))
        for i in range(batch_size):
            score_1[i] = sum_torch_list([word_tag_value[i][j] for j in range(word_size)])
            score_2[i] = sum_torch_list(
                [self.translation[path_tag[i][j]][path_tag[i][j + 1]] for j in range(word_size - 1)])
            score_2[i] += (self.translation[0][path_tag[i][0]] + self.translation[path_tag[i][-1]][self.num_type + 1])
            score[i] = score_1[i] + score_2[i]
        return score  # [batch_size, 1]

    def _score_total(self, word_tag_value):
        # word_tag_value : [batch_size, word_size, num_type]
        batch_size, word_size, num_type = word_tag_value.size()
        score_now = torch.zeros((batch_size, num_type + 2))  # [batch_size, num_type + 2]
        for i in range(batch_size):
            for j in range(1, num_type + 1):
                score_now[i][j] = word_tag_value[i][0][j - 1]

        for i in range(batch_size):
            for j in range(1, word_size):
                obs_word = word_tag_value[i][j]
                previous_word = score_now[i]  # [num_type + 2]
                obs_word = obs_word.view(1, -1).expand(num_type + 2, num_type + 2)
                previous_word = previous_word.view(1, -1).expand(num_type + 2, num_type + 2).t()
                score_matrix = previous_word + obs_word + self.translation
                score_now[i] = torch.logsumexp(score_matrix, dim=0)

        return torch.logsumexp(score_now, dim=1)  # [batch_size, 1]

    def _viterbi(self, word_tag_value):
        batch_size, word_size, num_type = word_tag_value.size()
        path = torch.empty_like(word_tag_value)
        scores = torch.empty_like(word_tag_value).fill_(-1000000)
        tag_path = []
        for i in range(batch_size):
            for type in range(1, num_type + 1):
                scores[i][0][type - 1] = word_tag_value[i][0][type - 1] + self.translation[0][type]

            for j in range(1, word_size):
                for this_type in range(1, num_type + 1):
                    for last_type in range(1, num_type + 1):
                        curr_score = word_tag_value[i][j][this_type] + self.translation[last_type][this_type] + \
                                     scores[i][j - 1][last_type - 1]
                        if curr_score >= scores[i][j][this_type - 1]:
                            scores[i][j][this_type - 1] = curr_score
                            path[i][j][this_type - 1] = last_type - 1

            max_score = -1000000
            curr_type = -1
            for type in range(1, num_type + 1):
                scores[i][-1][type - 1] += self.translation[type][num_type + 1]
                if scores[i][-1][type - 1] > max_score:
                    max_score = scores[i][-1][type - 1]
                    curr_type = type - 1

            temp_path = []
            for j in range(1, word_size):
                temp_path.append(curr_type)
                curr_type = path[i][-j][curr_type]
            temp_path.append(curr_type)
            temp_path.reverse()
            tag_path.append(temp_path)

        return tag_path

    def forward(self, word_tag_value, real_tag, output_loss):
        # batch_size, word_size, num_type = word_tag_value.size()
        if output_loss is True:
            return self._score_total(word_tag_value) - self._score_path(real_tag, word_tag_value)
