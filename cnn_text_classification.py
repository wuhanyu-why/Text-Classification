import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Input, Model
from keras.layers import Dense, Embedding, Conv2D, MaxPooling2D, Concatenate, Reshape, Flatten
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

# 加载预训练词向量(load the pre_trained word embedding)
def load_word_embedding(filepath, word_index, embedding_dim):
    count = 0
    embeddings_index = {}
    f = open(filepath, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    for i, word in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            count += 1
            embedding_matrix[i] = embedding_vector
    print(count)
    return embedding_matrix

# 定义参数(parameters)
embedding_dim = 100
max_seq_len = 400
num_filters = 256

def load_data():
    data_train = pd.read_csv("E:/datasets/ag_news_csv/train.csv", header=None)
    data_test = pd.read_csv("E:/datasets/ag_news_csv/test.csv", header=None)
    train_labels = np.array(data_train[0])
    test_labels = np.array(data_test[0])
    # 类别分别是1,2,3,4,转换成0,1,2,3
    train_labels = train_labels - 1
    test_labels = test_labels - 1
    Y_train = to_categorical(train_labels, num_classes=4)
    Y_test = to_categorical(test_labels, num_classes=4)
    # print(labels)
    X_train = np.array(data_train[2])
    X_test = np.array(data_test[2])
    print(X_train)
    print(X_train.shape)
    print("*" * 40)
    print(X_test.shape)
    tokenizer = Tokenizer(num_words=100000)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(X_train, maxlen=max_seq_len, padding="pre")
    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=max_seq_len, padding='pre')
    word_index = tokenizer.index_word
    return X_train, X_test, Y_train, Y_test, index_word

X_train, X_test, Y_train, Y_test, index_word = load_data()

embedding_matrix = load_word_embedding("E:/BaiduNetdiskDownload/glove.6B/glove.6B.100d.txt", index_word, embedding_dim)
print(embedding_matrix.shape)



# 定义模型(model)
seq_input = Input(shape=(max_seq_len, ), dtype='int32')
embedding = Embedding(input_dim=len(index_word) + 1,
                            output_dim=embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_seq_len,
                            trainable=False)(seq_input)
reshape = Reshape((max_seq_len, embedding_dim, 1))(embedding)
convs = []
filter_sizes = [3, 4, 5]
for filter_size in filter_sizes:
    conv = Conv2D(filters=num_filters,
                  kernel_size=(filter_size, embedding_dim),
                  padding='valid',
                  kernel_initializer='normal',
                  activation='relu')(reshape)
    pool = MaxPooling2D(pool_size=(max_seq_len-filter_size+1, 1),
                        strides=(1,1),
                        padding='valid')(conv)
    convs.append(pool)

concate_tensor = Concatenate(axis=1)(convs)

flatten = Flatten()(concate_tensor)

# 可以增加Dropout层防止过拟合，此处省略(we can add a dropout layer, but omitted here)
output = Dense(units=4, activation='softmax')(flatten)

model = Model(inputs=seq_input, outputs=output)
print(model.summary())

# plot_model(model, to_file="cnn_model.png", show_shapes=True, show_layer_names=False, rankdir='TB')
# plt.figure(figsize=(10, 10))
# image = plt.imread("cnn_model.png")
# plt.imshow(image)
# plt.axis("off")
# plt.show()

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=X_train, y=Y_train, batch_size=32, epochs=5)
loss, accuracy = model.evaluate(X_test, Y_test)
print("test_loss:", loss)
print("test_accuracy:", accuracy)
