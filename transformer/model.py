import math
import torch.nn as nn
import torch
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.5):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x):
        # [batch_size, seq_len, ebd_dim]
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    attn_pad_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch, 1, len_k]
    '''
    [[0 0 0 1 1],
     [0 0 0 1 1],
     [0 0 0 1 1]]
    对 seq_q 求相对于 seq_k 的注意力权值 V 时所做的 attention_mask
    对于 seq_q 中的 <PAD>, 我们将其视作正常值
    '''
    return attn_pad_mask.expand(batch_size, len_q, len_k)  # [batch, len_q, len_k]


def get_attn_subsequence_mask(seq):
    batch_size, tgt_len = seq.size()
    attn_subsequence_mask = np.triu(np.ones(shape=[batch_size, tgt_len, tgt_len]), k=1)
    attn_subsequence_mask = torch.from_numpy(attn_subsequence_mask).byte()
    '''
    [[0 1 1 1],
     [0 0 1 1],
     [0 0 0 1],
     [0 0 0 0]]
    为了将训练的串行化改为并行化而对 label 做的 mask
    要注意的是最后一行得出的结果不需要参与统计
    '''
    return attn_subsequence_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size, n_heads, len_q, d_k]
        :param K: [batch_size, n_heads, len_q, d_k]
        :param V: [batch_size, n_heads, len_v(=len_k), d_v]
        :param attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / np.sqrt(Q.size()[3])  # [batch_size, n_heads, len_q, len_q]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

    def forward(self, input_q, input_k, input_v, attn_mask):
        """
        Generally, input_k equals to input_v
        :param input_q: [batch_size, len_q, d_model]
        :param input_k: [batch_size, len_k, d_model]
        :param input_v: [batch_size, len_v, d_model]
        :param attn_mask: [batch_size, seq_len, seq_len]
        :return:
        """
        ori_input, batch_size = input_q, input_q.size(0)
        Q = self.W_Q(input_q).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = self.W_K(input_k).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.W_V(input_v).view(batch_size, -1, self.n_heads, self.d_v).permute(0, 2, 1, 3)
        attn_mask = attn_mask.unsqeeze(1).repeat(1, self.n_heads, 1, 1)  # [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.permute(0, 2, 1).reshape(batch_size, -1, self.n_heads * self.d_v)
        return self.ln(self.fc(context) + ori_input)


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_f):
        super(FeedForwardLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_f, bias=False),
            nn.ReLU(),
            nn.Linear(d_f, d_model, bias=False)
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.ln(x + self.fc(x))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_f):
        super(EncoderLayer, self).__init__()
        self.self_attn_layer = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.feed_forward_layer = FeedForwardLayer(d_model, d_f)

    def forward(self, x, attn_mask):
        x = self.self_attn_layer(x, x, x, attn_mask)
        x = self.feed_forward_layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_layers, n_heads, d_k, d_v, d_f):
        super(Encoder, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = PositionalEmbedding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v, d_f) for _ in range(n_layers)])

    def forward(self, inputs):
        outputs = self.emb(inputs)  # [batch, src_len, d_model]
        outputs = self.pos_emb(outputs)
        self_attn_mask = get_attn_pad_mask(inputs, inputs)  # [batch, src_len, src_len]
        for layer in self.layers:
            outputs = layer(outputs, self_attn_mask)
        return outputs


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_f):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.encoder_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.feed_forward_layer = FeedForwardLayer(d_model, d_f)

    def forward(self, decoder_inputs, encoder_outputs, self_attn_mask, encoder_attn_mask):
        """
        :param decoder_inputs: [batch_size, tgt_len, d_model]
        :param encoder_outputs: [batch_size, src_len, d_model]
        :param self_attn_mask: [batch_size, tgt_len, tgt_len]
        :param encoder_attn_mask: [batch_size, tgt_len, src_len]
        """
        decoder_outputs = self.self_attn(decoder_inputs, decoder_inputs, decoder_inputs,
                                         self_attn_mask)
        decoder_outputs = self.encoder_attn(decoder_outputs, encoder_outputs, encoder_outputs,
                                            encoder_attn_mask)
        decoder_outputs = self.feed_forward_layer(decoder_outputs)
        return decoder_outputs


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, max_len, d_model, n_layers, n_heads, d_k, d_v, d_f):
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_k, d_v, d_f) for _ in range(n_layers)])

    def forward(self, decoder_inputs, encoder_inputs, encoder_outputs):
        """
        :param decoder_inputs: [batch_size, tgt_len]
        :param encoder_inputs: [batch_size, src_len]
        :param encoder_outputs: [batch_size, src_len, d_model]
        """
        decoder_outputs = self.emb(decoder_inputs)
        decoder_outputs = self.pos_emb(decoder_outputs)
        dec_self_attn_mask = get_attn_pad_mask(decoder_inputs, decoder_inputs)
        dec_self_sub_mask = get_attn_subsequence_mask(decoder_inputs)
        dec_self_mask = torch.gt(dec_self_sub_mask + dec_self_attn_mask, 0)
        dec_enc_mask = get_attn_pad_mask(decoder_inputs, encoder_inputs)
        for layer in self.layers:
            decoder_outputs = layer(decoder_outputs, encoder_outputs, dec_self_mask, dec_enc_mask)
        return decoder_outputs


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, src_len, d_model, enc_layers, tgt_vocab_size, tgt_len, dec_layers, n_heads, d_k,
                 d_v, d_f):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, src_len, d_model, enc_layers, n_heads, d_k, d_v, d_f)
        self.decoder = Decoder(tgt_vocab_size, tgt_len, d_model, dec_layers, n_heads, d_k, d_v, d_f)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        '''
        :param enc_inputs: [batch_size, src_len]
        :param dec_inputs: [batch_size, tgt_len]
        '''
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs)  # [batch_size, tgt_len, d_model]
        outputs = self.projection(dec_outputs[:, :-1, :])
        # [batch_size, tgt_len - 1, tgt_vocab_size], 最后一列不算入答案内, dec_outputs[:,:-1,:]对应 label[:,1:,:]
        return outputs
