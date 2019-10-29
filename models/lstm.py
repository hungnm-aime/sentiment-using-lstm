# _*_ coding: utf-8 _*_
"""IMPORT LIB"""
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pickle as pkl

"""IMPORT MY FUNC"""
from embeddings.builder import glove_load, create_weights_matrix, create_emb_layer
from models.data_loader import train_loader, valid_loader, test_loader

glove = glove_load()
with open('../dataset/vocab.pkl', 'rb') as f:
    word2idx, idx2word = pkl.load(f)
weights_matrix = create_weights_matrix(word2idx, idx2word, glove)
weights_matrix = torch.from_numpy(weights_matrix)
embedding_layer, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, False)

batch_size = 128
print_every = 100
step = 0
n_epochs = 4  # validation loss increases from ~ epoch 3 or 4
# clip = 5  # for gradient clip to prevent exploding gradient problem in LSTM/RNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_size = 500
num_layer = 1
out_features = 1


class SentimentLSTM(nn.Module):

    def __init__(self, _embedding_layer, _embedding_dim, _hidden_size, _num_layers, _out_features):
        super(SentimentLSTM, self).__init__()
        self.hidden_size = _hidden_size
        self.num_layers = _num_layers
        self.embedding_layer = _embedding_layer

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=_hidden_size,
                            num_layers=_num_layers,
                            batch_first=True)

        self.fc = nn.Linear(in_features=_hidden_size, out_features=_out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_words):
        embedded_words = self.embedding_layer(input_words)  # (batch_size, seq_length, embedding_dim)
        lstm_out, h = self.lstm(embedded_words)  # (batch_size, seq_length, hidden_size)
        # lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)  # (batch_size*seq_length, hidden_size)
        fc_out = self.fc(lstm_out)  # (batch_size*seq_length, n_output)
        sigmoid_out = self.sigmoid(fc_out)  # (batch_size*seq_length, n_output)
        sigmoid_out = sigmoid_out.view(batch_size, -1)  # (batch_size, seq_length*n_output)
        sigmoid_out = sigmoid_out.view(-1,  8*out_features)  # (batch_size, seq_length*n_output)

        # extract the output of ONLY the LAST output of the LAST element of the sequence
        sigmoid_last = sigmoid_out[:, -1]  # (batch_size, 1)

        return sigmoid_last, h

    def init_hidden(self, _batch_size):
        # if self.bidirectional:
        #     return torch.zeros((2, batch_size, self.n_hidden)).to(device)
        # else:
        #     return torch.zeros((1, batch_size, self.n_hidden)).cuda().to(device)
        return torch.zeros((1, _batch_size, self.hidden_size)).to(device)


# n_vocab = len(vocab_to_int)
# n_embed = 400
# n_hidden = 512
# n_output = 1   # 1 ("positive") or 0 ("negative")
# n_layers = 1

net = SentimentLSTM(_embedding_layer=embedding_layer,
                    _embedding_dim=embedding_dim,
                    _hidden_size=hidden_size,
                    _num_layers=num_layer,
                    _out_features=out_features)
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in range(n_epochs):
    h = net.init_hidden(batch_size)

    for inputs, labels in train_loader:
        step += 1
        inputs, labels = inputs.to(device), labels.to(device)

        # making requires_grad = False for the latest set of h
        h = tuple([each.data for each in h])

        net.zero_grad()
        output, h = net(inputs)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # nn.utils.clip_grad_norm(net.parameters(), clip)
        optimizer.step()

        if (step % print_every) == 0:
            """VALIDATION"""
            net.eval()
            valid_losses = []
            v_h = net.init_hidden(batch_size)

            for v_inputs, v_labels in valid_loader:
                v_inputs, v_labels = inputs.to(device), labels.to(device)

                v_h = tuple([each.data for each in v_h])

                v_output, v_h = net(v_inputs)
                v_loss = criterion(v_output.squeeze(), v_labels.float())
                valid_losses.append(v_loss.item())

            print("Epoch: {}/{}".format((epoch + 1), n_epochs),
                  "Step: {}".format(step),
                  "Training Loss: {:.4f}".format(loss.item()),
                  "Validation Loss: {:.4f}".format(np.mean(valid_losses)))
            net.train()
