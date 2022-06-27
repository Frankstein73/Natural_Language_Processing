import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, num_embeddings, length, embedding_size, feature_size, kernel_size, dropout, num_type):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_size)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embedding_size, out_channels=feature_size, kernel_size=k_size),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=length - k_size + 1)
            )
            for k_size in kernel_size
        ])
        self.fc = nn.Linear(feature_size * len(kernel_size), num_type)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, length, embedding_dim]
        x = x.permute(0, 2, 1)  # [batch_size, embedding_dim, length]
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x, 1)
        x = x.view(-1, x.size(1))  # [batch_size, feature_size * len(kernel_size)]
        x = self.dropout(x)
        x = self.fc(x)
        return x
