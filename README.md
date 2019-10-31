## Sử dụng Pytorch  xây dựng mạng LSTM với word embedding (**glove**) để phân loại review IMDB.

###Các bước thực hiện trong hướng dẫn này:
1. Khám phá data
2. Tiền xử lý
3. Padding/truncating data
4. Chia data thành train, valid, test
5. Dataloader và Batching
6. Kiến trúc LSTM
7. Coding Model
8. Training Network
9. Testing

##

###1. Khám phá data.
Data sử dụng để thực hành trong hướng dẫn này được lấy từ tập dataset movie của [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/).
Data được chia thành 2 file: *imdb.neg.txt*, *imdb.pos.txt* tương ứng với 2 nhãn **positive** và **negative**.

Xây dựng hàm đọc dữ liệu từ 2 file này:
*imdb.neg.txt*, *imdb.pos.txt* được đặt trong thư mục dataset

```Code nằm trong dataset/reader.py```
```python
def read_dataset(path):
    with open(path) as f:
        X = f.readlines()
    return X

X_neg = read_dataset('dataset/imdb.neg.txt')
print("number example of negative: ", len(X_neg))
X_pos = read_dataset('dataset/imdb.pos.txt')
print("number example of positive: ", len(X_pos))
```

**output**
```
number example of negative: 30000
number example of positive: 30000
```

Như vậy số lượng nhãn neg và pos đều tương đối lớn và không bị mất cân bằng.

###2. Tiền xử lý data.
Trong bước này sẽ lần lượt thực hiện các bước.

####2.1 Làm sạch data.
Bao gồm tách từ và xóa bỏ các kí tự đặc biệt.

```Code nằm trong thư mục dataset/reader```

```python
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

def clean_text(X, label):
    X_tmp = []
    for x in X:
        text = x.lower()
        text = tokenizer.tokenize(text)
        new_sentence = []
        for w in text:
            if w not in stop_words:
                new_sentence.append(w)
        if len(new_sentence) > 0:
            X_tmp.append(new_sentence)
    if label == 0:
        y = list(np.zeros((len(X_tmp),), dtype=int))
    else:
        y = list(np.ones((len(X_tmp),), dtype=int))
    return X_tmp, y
```
Hàm trên sẽ thực hiện làm sạch data đối với mỗi loại nhãn negative và positive, đồng thời khởi tạo nhãn **y** tương ứng.
Việc tokenize và loại bỏ stopword sử dụng thư viện **[nltk](https://pypi.org/project/nltk/)**. 
Do data tiếng anh nên chỉ cần tách từ theo khoảng trắng.

Sau đó thực hiện đọc dữ liệu pos và neg đồng thời shuffle.

```
    X_pos = read_dataset('./imdb.pos.txt')
    X_pos, y_pos = clean_text(X_pos, 1)

    X_neg = read_dataset('./imdb.neg.txt')
    X_neg, y_neg = clean_text(X_neg, 0)

    X = X_pos + X_neg
    y = y_pos + y_neg
    X, y = shuffle(X, y, random_state=0)


    with open('./train_raw.pkl', 'wb') as f:
        pkl.dump([X, y], f)

    with open('./train_raw.pkl', 'rb') as f:
        X, y = pkl.load(f)

    print(X)
```

Đoạn code trên có thực hiện lưu dữ liệu X, y sau khi xử lý để tránh thời gian xử lý lần sau.

#####Xây dựng từ điển word2idx và idx2word dựa trên tập từ điển của dataset.
```
def build_vocab(X):
    vocab = Counter()
    for x in X:
        vocab.update(x)
    word2idx = {_: i + 2 for i, _ in enumerate(vocab)}
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    return word2idx, idx2word
```

Sau đó lưu lại 

```
    w2i, i2w = build_vocab(X)
    print(w2i)
    with open('./vocab.pkl', 'wb') as f:
        pkl.dump([w2i, i2w], f)
```

Cuối cùng của bước tiền xử lý đó là convert các từ thành các index tương ứng để phục vụ cho tầng embedding.

```
def save_x_y(X, y, w2i):
    """
    lưu X và y dạng index
    :return:
    """
    X_tmp = []
    for x in X:
        X_tmp.append([w2i[w] for w in x])
    with open('../dataset/x_y_index.pkl', 'wb') as f:
        pkl.dump([X_tmp, y], f)
```


###3. Padding/truncating data
##

Do các đánh giá có độ dài ngắn khác nhau, để giải quyết vấn đề này sẽ sử dụng padding và truncating.
Cụ thể cần xác định độ dài của câu cố định (sequence length), với những câu dài hơn sequence length thì sẽ truncating,
ngược lại với các văn bản ngắn hơn seqence length thì sử dụng padding các giá trị 0 ở đầu chuỗi.

``Code đọc data đã xử lý ở bước 1``
```
def read_X_y():
    with open('../dataset/x_y_index.pkl', 'rb') as f:
        X, y = pkl.load(f)
    return X, y
```


```
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
```

```X, y = read_X_y()
   X = pad_features(X=X, seq_length=8)
   print(X[100)
```

###4. Tập train, valid, test
##

Đoạn code dưới đây chia tập data ban đầu thành tập train, valid, test tương ứng với tỉ lệ 8:1:1
```
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

```

###5. Dataloader và Batching
##
Sau khi tạo tập train, valid, test tiếp theo là tạo dataloader cho tập data này

```
def create_data_loader(train_x, train_y, valid_x, valid_y, test_x, test_y, batch_size=64):
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # dataloaders
    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return train_loader, valid_loader, test_loader
```

```
X, y = read_X_y()
X = pad_features(X=X, seq_length=8)

train_x, train_y, valid_x, valid_y, test_x, test_y = split_data(X, y)

train_loader, valid_loader, test_loader = create_data_loader(train_x, train_y, valid_x, valid_y, test_x, test_y, batch_size=128)

```

###6. Kiến trúc LSTM
##
![Kiến trúc LSTM để phân loại văn bản](imgs/lstm.png)

Các layer trong kiến trúc:
0. Tokenize: Bước này thực chất không phải là 1 layer trong kiến trúc mạng bởi chúng ta đã thực hiện trong bước tiền xử lý.
Bước tiền xử lý thực hiện tách từ và convert mỗi token về giá trị integer. 

1. Embedding layer: thực hiện convert mỗi giá trị integer sang embedding có kích thước xác định.
Cụ thể tầng này sử dụng pre-train glove với kích thước embedding 50, có thể tải glove [tại đây](http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip)

Do embedding của glove là file text, mỗi dòng bao gồm 1 từ và 1 vector tương ứng. Vậy cần xây dựng 1 module với input là word và output sẽ là vector tương ứng 
(giống như cách thư viện [gensim](https://radimrehurek.com/gensim/models/keyedvectors.html) thực hiện)

*Code dưới đây thực hiện build glove*

```python

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
```

```python
def glove_load():
    vectors = bcolz.open(f'../dataset/6B.50.dat')[:]
    words = pkl.load(open(f'../dataset/6B.50_words.pkl', 'rb'))
    word2idx = pkl.load(open(f'../dataset/6B.50_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    return glove
```

Sau đó tạo ma trận trọng số như dưới đây
```python 
def create_weights_matrix(w2i, i2w, glove):
    matrix_len = len(w2i)
    weights_matrix = np.zeros((matrix_len, EMBEDDING_DIM))
    for i, w in i2w.items():
        try:
            weights_matrix[i] = glove[w]
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))
    return weights_matrix
```

Các params input:
* w2i: bộ word2index mà đã xây dựng trong bước tiền xử lý 
* i2w: bộ index2word cũng được xây dựng trong bước tiền xử lý. 
* glove: kết quả trả về của hàm ``glove_lod``

Cuối cùng là xây dựng layer embedding: 
```python 
def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape[0], weights_matrix.shape[1]
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

```

2. LSTM layer: được xác định bởi các params: 
* ```input_size```: chính là output của tầng embedding, cụ thể embedding_dim = 50
* ```hidden_size```: Kích thước của tầng hidden
* ```num_layers```: Số tầng LSTM stack lên nhau
 
3. Fully connected layer: tầng này thực hiện mapping output từ tầng LSTM đến kích thước đầu ra mong muốn

4. Sigmoid activation layer: convert các output sang khoảng 0-1.

5. Output: kết quả của sigmoid tại steptime cuối cùng được xem như là output cuối cùng.
 
 