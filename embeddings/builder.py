import numpy as np
import torch.nn as nn
import pickle as pkl
import torch
import bcolz

EMBEDDING_DIM = 50


def glove_build(path):
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'../dataset/6B.50.dat', mode='w')

    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = line.split()
            word = values[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(values[1:]).astype(np.float)
            vectors.append(vect)

    word = '<pad>'
    words.append(word)
    word2idx[word] = idx
    vect = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))
    vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400002, 50)), rootdir=f'../dataset/6B.50.dat', mode='w')
    vectors.flush()

    pkl.dump(words, open(f'../dataset/6B.50_words.pkl', 'wb'))
    pkl.dump(word2idx, open(f'../dataset/6B.50_idx.pkl', 'wb'))


def glove_load():
    vectors = bcolz.open(f'../dataset/6B.50.dat')[:]
    words = pkl.load(open(f'../dataset/6B.50_words.pkl', 'rb'))
    word2idx = pkl.load(open(f'../dataset/6B.50_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    return glove


def create_weights_matrix(w2i, i2w, glove):
    matrix_len = len(w2i)
    weights_matrix = np.zeros((matrix_len, EMBEDDING_DIM))
    for i, w in i2w.items():
        try:
            weights_matrix[i] = glove[w]
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))
    return weights_matrix


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape[0], weights_matrix.shape[1]
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


if __name__ == "__main__":
    # glove_build('/Users/manhhung/Documents/workspace/share-ai/classify/dataset/glove.6B/glove.6B.50d.txt')

    glove = glove_load()
    print(glove['<pad>'])

    with open('../dataset/vocab.pkl', 'rb') as f:
        w2i, i2w = pkl.load(f)

    weights_matrix = create_weights_matrix(w2i, i2w, glove)
    weights_matrix = torch.from_numpy(weights_matrix)
    emb_layer, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, False)
    print(emb_layer)
