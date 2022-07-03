import numpy as np
import torch


def data_process(word_dict, data):
    max_len = 0
    labels = []
    for item in data:
        sentence1, sentence2 = item['sentence1'], item['sentence2']
        max_len = max(max_len, len(sentence1.split()), len(sentence2.split()))
        label = item['gold_label']
        if label == '-':
            label = max(item['annotator_labels'], key=item['annotator_labels'].count)
        labels.append({'entailment': 0, 'neutral': 1, 'contradiction': 2}[label])

    features = np.zeros((len(data), 2, max_len), dtype=np.int64)
    for i, item in enumerate(data):
        strs = [item['sentence1'], item['sentence2']]
        for sentence_index in range(2):
            for j, phrase in enumerate(strs[sentence_index].split()):
                try:
                    features[i][sentence_index][j] = word_dict.stoi[
                        ''.join(filter(str.isalpha, phrase.lower().strip()))]
                except:
                    pass
    features = torch.from_numpy(features)
    labels = torch.tensor(labels)
    return features, labels
