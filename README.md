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