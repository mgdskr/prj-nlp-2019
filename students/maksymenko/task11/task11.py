def split_tags(string):
    return [tuple(i.split("/")) for i in string.split()]

def readTrainData(filename):
    data = []
    for line in open(filename):
        line = line.strip()
        #read in training or dev data with labels
        if len(line.split('\t')) == 7:
            (trendid, trendname, origsent, candsent, judge, origsenttag, candsenttag) = \
            line.split('\t')
        else:
            continue
        # ignoring the training data that has middle label
        nYes = eval(judge)[0]
        if nYes >= 3:
            amt_label = True
            data.append((split_tags(origsenttag), split_tags(candsenttag), amt_label))
        elif nYes <= 1:
            amt_label = False
            data.append((split_tags(origsenttag), split_tags(candsenttag), amt_label ))
    return data

def readTestData(filename):
    data = []
    for line in open(filename):
        line = line.strip()
        #read in training or dev data with labels
        if len(line.split('\t')) == 7:
            (trendid, trendname, origsent, candsent, judge, origsenttag, candsenttag) = \
            line.split('\t')
        else:
            continue
        # ignoring the training data that has middle label
        nYes = int(judge[0])
        if nYes >= 4:
            expert_label = True
        elif nYes <= 2:
            expert_label = False
        else:
            expert_label = None
        data.append((split_tags(origsenttag), split_tags(candsenttag), expert_label))
    return data

dir = 'students/maksymenko/task11/'

train_data = readTrainData(dir + 'SemEval-PIT2015-py3/data/dev.data')
test_data = readTestData(dir + 'SemEval-PIT2015-py3/data/test.data')


from gensim.models import KeyedVectors

from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

PATH = 'students/maksymenko/task11/numberbatch-en-17.06.txt'
word_vectors = KeyedVectors.load_word2vec_format(PATH, encoding='utf-8', binary=False)


def get_sentence_vector(sentence):
    sentence_length = 0
    sentence_vector = 0
    for word in sentence:
        word_text = word[0].lower()
        meta_info = word[1:]
        try:
            sentence_vector += word_vectors[word_text]
            sentence_length += 1
        except:
            pass
    return sentence_vector / sentence_length


def prepare_data(data):
    X = []
    y = []
    for line in data:
        sentence_1, sentence_2, label = line
        if label == None:
            continue
        vector_1 = get_sentence_vector(sentence_1)
        vector_2 = get_sentence_vector(sentence_2)
        vec1 = np.array(vector_1)
        vec2 = np.array(vector_2)

        cos_sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1), dense_output=True)
        # almost the same result as just for cos similarity
        # vector = vector_1.tolist() + vector_2.tolist() + cos_sim[0].tolist()
        vector = cos_sim[0].tolist()
        X.append(vector)
        y.append(label.__repr__())

    return X, y


X_train, y_train = prepare_data(train_data)
X_test, y_test = prepare_data(test_data)

clf = LogisticRegression(
    random_state=0,
    solver='lbfgs',
    multi_class='auto',
    max_iter=500
).fit(X_train, y_train)

clf.score(X_test, y_test)
# ~83%
