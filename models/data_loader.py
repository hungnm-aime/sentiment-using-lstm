
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def read_X_y():
    with open('../dataset/x_y_index.pkl', 'rb') as f:
        X, y = pkl.load(f)
    return X, y


def pad_features(X, seq_length):
    X_new = np.zeros((len(X), seq_length), dtype=int)

    for i, x in enumerate(X):
        len_x = len(x)
        if len_x < seq_length:
            zeros = list(np.zeros(seq_length - len_x))
            new = zeros + x
        elif len_x > seq_length:
            new = x[:seq_length]
        else:
            new = x
        X_new[i, :] = np.array(new)
    return X_new


# X, y = read_X_y()
# X = pad_features(X=X, seq_length=10)
# print(X[300])


# Chia dữ liệu thành tập train, valid, test
def split_data(features, encoded_labels):
    encoded_labels = np.array(encoded_labels)
    len_feat = len(features)
    split_frac = 0.8
    train_x = features[0:int(split_frac * len_feat)]
    train_y = encoded_labels[0:int(split_frac * len_feat)]
    remaining_x = features[int(split_frac * len_feat):]
    remaining_y = encoded_labels[int(split_frac * len_feat):]
    valid_x = remaining_x[0:int(len(remaining_x) * 0.5)]
    valid_y = remaining_y[0:int(len(remaining_y) * 0.5)]
    test_x = remaining_x[int(len(remaining_x) * 0.5):]
    test_y = remaining_y[int(len(remaining_y) * 0.5):]

    return train_x, train_y, valid_x, valid_y, test_x, test_y


# Dataloaders and Batching


def create_data_loader(train_x, train_y, valid_x, valid_y, test_x, test_y, batch_size=64):
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # dataloaders
    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()
    print('Sample input size: ', sample_x.size())  # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size())  # batch_size
    print('Sample label: \n', sample_y)
    return train_loader, valid_loader, test_loader


X, y = read_X_y()
X = pad_features(X=X, seq_length=8)

train_x, train_y, valid_x, valid_y, test_x, test_y = split_data(X, y)

train_loader, valid_loader, test_loader = create_data_loader(train_x, train_y, valid_x, valid_y, test_x, test_y, batch_size=128)


if __name__ == "__main__":
    X, y = read_X_y()
    X = pad_features(X=X, seq_length=8)

    train_x, train_y, valid_x, valid_y, test_x, test_y = split_data(X, y)

    train_loader, valid_loader, test_loader = create_data_loader(train_x, train_y, valid_x, valid_y, test_x, test_y, batch_size=64)
