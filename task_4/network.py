import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from opti import *


class BiLstmCRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, tag_to_index, hidden_size, max_length, dropout=0.5):
        super(BiLstmCRF, self).__init__()
        self.ebd = nn.Embedding(vocab_size, embedding_dim, padding_idx=-2)
        torch.nn.init.kaiming_uniform_(self.ebd.weight)
        self.hidden_size = hidden_size
        self.ebd_size = embedding_dim
        self.tag_to_index = tag_to_index
        self.num_tag = len(tag_to_index)
        self.max_len = max_length
        self.lstm = nn.LSTM(input_size=self.ebd_size, hidden_size=hidden_size, bidirectional=True)
        self.hidden_to_tag = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, self.num_tag),
            # nn.ReLU(),
            
        )
        self.crf = CRF(self.num_tag, batch_first=True)

    def _mask_process(self, length_list):
        mask = []
        for length in length_list:
            mask.append([1 for i in range(length)] + [0 for j in range(self.max_len - length)])
        return torch.tensor(mask, dtype=torch.bool).to(device)

    def _lstm_process(self, sentences, length_list):
        ebd = self.ebd(sentences)
        packed_sentences = pack_padded_sequence(ebd, lengths=length_list.to('cpu'), batch_first=True,
                                                enforce_sorted=False)
        res, _ = self.lstm(packed_sentences)
        res, _ = pad_packed_sequence(res, batch_first=True, total_length=self.max_len)
        return self.hidden_to_tag(res)

    def _crf_process(self, sentences, tags, length_list):
        return self.crf(sentences, tags, self._mask_process(length_list))

    def predict(self, sentences, length_list):
        return self.crf.decode(self._lstm_process(sentences, length_list), self._mask_process(length_list))

    def forward(self, sentences, tags, length_list):
        x = self._lstm_process(sentences, length_list)
        x = self._crf_process(x, tags, length_list)
        return x
