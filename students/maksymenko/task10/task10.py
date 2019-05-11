import os
import re
from gensim.models import KeyedVectors
from nltk import word_tokenize
from langdetect import detect
from sklearn.model_selection import train_test_split

directory = 'students/maksymenko/task10/1551'
PATH = 'students/maksymenko/task10/news.lowercased.tokenized.word2vec.300d'
word_vectors = KeyedVectors.load_word2vec_format(PATH, encoding='utf-8', binary=False)


def get_paragraph_vector(tokens):
    paragraph_embedding = 0
    tokens_num = 1
    for token in word_tokenize(tokens):
        token = token.lower()
        try:
            paragraph_embedding += word_vectors[token]
            tokens_num += 1
        except:
            pass
    return paragraph_embedding / tokens_num


filenames = []
texts = []
vectors = []
labels = []
labels_idxs = []


def label_to_idx(label):
    return filenames.index(label)


for filename in os.listdir(directory):
    with open(directory + '/'+ filename, 'r', encoding='utf-8') as file:
        filenames.append(filename)
        print(filename)
        paragraphs = re.split(r'\n\d{7}\n', file.read())
        for par in paragraphs:
            try:
                if detect(par) != 'uk':
                    continue
            except:
                continue
            cleaned_par = re.sub(r'\n', ' ', par)
            vector = get_paragraph_vector(cleaned_par)
            if type(vector) != float and len(cleaned_par) > 0:
                labels.append(filename)
                labels_idxs.append(label_to_idx(filename))
                texts.append(cleaned_par)
                vectors.append(vector)

print(len(vectors), len(labels_idxs))


X_train_arr, X_test_arr, y_train_arr, y_test_arr = train_test_split(vectors, labels_idxs, test_size=0.1, random_state=42)


import tensorflow as tf
import numpy as np

print_every = 100
learning_rate = 1e-2


def check_accuracy(sess, dset, x, scores, is_training=None):
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        feed_dict = {x: x_batch, is_training: 0}
        scores_np = sess.run(scores, feed_dict=feed_dict)
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    print(f'Got {num_correct} / {num_samples} correct ({acc * 100}%)')


class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
            self.X, self.y = self.X[idxs], self.y[idxs]
        return iter((self.X[i:i + B], self.y[i:i + B]) for i in range(0, N, B))



def train(train_dset, val_dset, model_init_fn, optimizer_init_fn, num_epochs=1):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 300])
    y = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.bool, name='is_training')

    scores = model_init_fn(x)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
    loss = tf.reduce_mean(loss)

    optimizer = optimizer_init_fn()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t = 0
        for epoch in range(num_epochs):
            print(f'Starting epoch ${epoch}')
            for x_np, y_np in train_dset:
                feed_dict = {x: x_np, y: y_np, is_training: 1}
                loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                if t % print_every == 0:
                    print(f'Iteration %d, loss = %.4f' % (t, loss_np))
                    check_accuracy(sess, val_dset, x, scores, is_training=is_training)
                    print()
                t += 1


def model_init_fn(inputs):
    hidden_layer_size, num_classes = 10000, 188
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hidden_layer_size, input_shape=(None, 300)))
    model.add(tf.keras.layers.Activation('elu'))
    model.add(tf.keras.layers.Dense(num_classes))
    return model(inputs)


def optimizer_init_fn():
    return tf.train.GradientDescentOptimizer(learning_rate)


X_train = np.array(X_train_arr)
y_train = np.array(y_train_arr)
X_test = np.array(X_test_arr)
y_test = np.array(y_test_arr)

train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
val_dset = Dataset(X_test, y_test, batch_size=64, shuffle=False)
# test_dset = Dataset(X_test, y_test, batch_size=64)

train(train_dset, val_dset, model_init_fn, optimizer_init_fn, 100)

# 100 training epochs 58% - no improvement vs Logistic Regression(%60) 🤔