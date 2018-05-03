import spacy
import os
import sys
import keras
import pathlib
from collections import Counter
import time
from keras.callbacks import ModelCheckpoint
curdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curdir+"/../../")

from Utils.WordVecs import *
from Utils.MyMetrics import *
from Utils.Datasets import *
from Utils.Semeval_2013_Dataset import *
import lstm_bilstm



def parse(docs, lang):
    nlp = spacy.load(lang)
    start = time.time()
    nlp(docs[0],disable=['parser', 'tagger', 'ner'])
    end = time.time()
    diff = end - start
    print("Estimated runtime", diff*len(docs))
    ret = [nlp(doc, disable=['parser', 'tagger', 'ner']) for doc in docs]
    return ret

def train(sentences_train, sentences_test, train_labels, test_labels, lang):
    name = "SpanishMovieReview"
    bi = True
    vecs = WordVecs('../../embeddings/wiki.multi.'+lang+'.vec', 'word2vec')
    dim = vecs.vector_size
    vocab = {}
    output_dim = len(Counter(train_labels).keys())

    # parse data with spacy
    print("parsing training data")
    sentences_train_parsed = parse(sentences_train, lang)
    print("parsing test data")
    sentences_test_parsed = parse(sentences_test, lang)
    print("Done parsing data")

    max_length = 0
    for sent in sentences_train_parsed:
        max_length = len(sent) if len(sent) > max_length else max_length
        for w in sent:
            if w.text not in vocab:
                vocab[w.text.lower()] = 1
            else:
                vocab[w.text.lower()] += 1

    wordvecs = {}
    for w in vecs._w2idx.keys():
        if w in vocab:
            wordvecs[w] = vecs[w]

    train = False # don't change word embeddings
    W, word_idx_map = lstm_bilstm.get_W(wordvecs)
    # DANITER: Search in the future
    best_dim = 100
    best_dropout = .4
    best_epoch = 12
    best_f1 = 0.1


    def encode_sent(sent, word_idx_map, max_length=max_length):
        encoded = np.array([word_idx_map[w.text.lower()] for w in sent if w.text.lower() in word_idx_map])
        return encoded

    train_data = []
    for sent in sentences_train_parsed:
        train_data.append(encode_sent(sent, word_idx_map))
    train_data = lstm_bilstm.pad_sequences(train_data, max_length)

    test_data = []
    for sent in sentences_test_parsed:
        test_data.append(encode_sent(sent, word_idx_map))
    test_data = lstm_bilstm.pad_sequences(test_data, max_length)

    train_y = keras.utils.to_categorical(train_labels)
    test_y = keras.utils.to_categorical(test_labels)

    clf = lstm_bilstm.create_BiLSTM(wordvecs, best_dim, output_dim, best_dropout, weights=W, train=train)
    pathlib.Path('models/bilstm/' + name +'/run').mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint('models/bilstm/' + name +'/run/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

    hist = clf.fit(train_data, train_y, validation_data=[test_data, test_y],
                epochs=best_epoch, verbose=1, callbacks=[checkpoint])

    return (hist, clf)
