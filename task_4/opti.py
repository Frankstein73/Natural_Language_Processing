# const
import torch.cuda

# padding_idx = -1
# unknown_idx = -2
PATH = '../../python_data/data/conll2003en'
# opt
train_length = 50000
valid_length = 1000
batch_size_train = 10
batch_size_valid = 10
hidden_size = 200
ebd_dim = 50
learning_rate = 0.015
epochs = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
