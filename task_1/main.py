import pandas as pd
from module import Ngram
from module import SoftmaxRegression

PATH = '../../python_data/data/'
NUM_TYPE = 5

num_sample = 1000
per = 0.3  # data_train : data_valid == 7 : 3
ngram = 3
epochs = 1000000
learning_rate = 1
strategy = 'mini_batch'  # shuffle or mini_batch or batch
mini_batch_size = 100

if __name__ == '__main__':
    data = pd.read_csv(PATH + 'train.tsv', sep='\t')
    bag = Ngram(data=data, ngram=range(1, ngram + 1), iters=num_sample)
    X_train, y_train, X_valid, y_valid = bag.data_split(per=per)
    reg = SoftmaxRegression(num_type=NUM_TYPE)
    reg.fit(X_train, y_train, lr=learning_rate, epochs=epochs, strategy=strategy, mini_size=mini_batch_size)
    correct_rate_train = reg.predict(X_train, y_train)
    correct_rate_valid = reg.predict(X_valid, y_valid)
    print('Train:', correct_rate_train)
    print('Valid:', correct_rate_valid)
    # reg.loss_plot()
