# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import gensim
from utils import task_to_df
# ---------------------------------------------------------------------------- Utility scripts
# setup logger
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
                    handlers=[logging.StreamHandler()])


def load_w2v_model(w2v_file, normalized_w2v=False):
    model_fname = "./input/models/google_w2v.model"
    model_fname_norm = "./input/models/google_w2v_norm.model"

    if not Path(model_fname).exists():
        w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=True)
        w2v.save(model_fname)
        # also save normalize model
        w2v.init_sims(
            replace=True)  # replace = true mean keep only normalized vectors, remove original, after removal further training is not possible
        w2v.save(model_fname_norm)

    if normalized_w2v == True:
        return gensim.models.KeyedVectors.load(model_fname_norm, mmap='r')
    else:
        return gensim.models.KeyedVectors.load(model_fname, mmap='r')


def get_w2v_word_embeddings(words, normalized_w2v=False):
    w2v = load_w2v_model(None, normalized_w2v)
    vecs = np.array([])
    for word in words:

        if type(word) == float and np.isnan(word) == True:
            v = np.zeros(300).reshape(1, 300)
        else:
            w = word.split('_')[0]  # in semeval dataset _ is used as connector for multi-words verb
            if not w in w2v.index2word:
                v = np.zeros(300).reshape(1, 300)
            else:
                v = w2v.get_vector(w).reshape(1, 300)

        if vecs.size == 0:
            vecs = v
        else:
            vecs = np.vstack((vecs, v))
    return vecs


import pickle


def load_tfidf(tfidf_file, vocabulary):
    if not '.pkl' in tfidf_file:
        tfidf_dump = '{}_tdfidfvec.pkl'.format((tfidf_file).split('.txt')[0])
    else:
        tfidf_dump = tfidf_file

    if Path(tfidf_dump).exists():
        f = open(tfidf_dump, 'rb')
        tfidf_vec = pickle.load(f)
        f.close()

    else:
        logging.info('{} not found. Fitting vectorizer on {}'.format(tfidf_dump, tfidf_file))
        tfidf_vec = TfidfVectorizer(vocabulary=vocabulary)
        fp = open(tfidf_file, 'r', encoding='utf-8')
        tfidf_corpus = fp.readlines()
        tfidf_vec.fit(tfidf_corpus)
        f = open(tfidf_dump, 'wb')
        pickle.dump(tfidf_vec, f, -1)
        f.close()

    return tfidf_vec


# to calculate power of tfidf and chi2
def weight_power(Q, p):
    if type(Q) == scipy.sparse.csr.csr_matrix:
        return Q.power(p)
    else:
        return np.power(Q, p)


# tokenize context, taken from russe
def pre_process(df):

    df.context = df.context.str.lower()
    return df

# ------------------------------------------------------
def get_w2v_context_embeddings(df, w2v_file, tfidf_file=None, normalized_w2v=False, normalize_weights="l2",
                               normalize_avg="l2", weighting_scheme="tfidf", tfidf_power=1):
    # global logging
    logging.info('------------------------------------------')
    logging.info('w2v_file = ' + w2v_file)
    logging.info('tfidf_file = ' + tfidf_file)

    logging.info('normalized_w2v = ' + str(normalized_w2v))
    logging.info('normalize_weights = ' + str(normalize_weights))
    logging.info('normalize_avg = ' + str(normalize_avg))

    logging.info('weighting_scheme = ' + str(weighting_scheme))
    if weighting_scheme == "tfidf" or weighting_scheme == "tfidf_chi2":
        logging.info('tfidf = ' + str(tfidf_power))

    logging.info('------------------------------------------')
    # ----------------------------------------- load w2v, input data and do preprocessing if needed
    # --------------- load Google's pretrained model
    w2v = load_w2v_model(w2v_file, normalized_w2v)
    # --------------- pre-processing
    df = pre_process(df)
    # ----------------------------------------- calculate weights e.g. tfidf
    # --------------- TF_IDF
    tfidf_vec = TfidfVectorizer(vocabulary=w2v.index2word)
    if tfidf_file:
        logging.info('using external data for tfidf')
        tfidf_vec = load_tfidf(tfidf_file, w2v.index2word)
        TF_IDF = tfidf_vec.transform(df['context'])
    else:
        logging.info('using internal data for tfidf')
        TF_IDF = tfidf_vec.fit_transform(df['context'])  # Returns tf-idf-weighted document-term matrix, size = rows_in_df by vocabulary_


    W = weight_power(TF_IDF, tfidf_power)
    # ---------------  W is final term-weight matrix
    if normalize_weights:
        W = normalize(W, norm=normalize_weights, axis=1, copy=False)
    # ----------------------------------------- get vectors for sentences
    context_vecs = W.dot(w2v.vectors)  # get vectors for all sentences, size = rows_in_df by w2v.vector_size
    sum_W = np.sum(W, axis=1)
    sum_W[np.where(sum_W == 0)] = 1  # if sentence was not found in w2v vocabulary, its sumOfWeights will be 0

    context_vecs = context_vecs / sum_W  # average out

    if normalize_avg:
        context_vecs = sklearn.preprocessing.normalize(context_vecs, norm=normalize_avg, axis=1)

    return context_vecs

def get_w2v_context_embeddings_Default(task, dataset):
    w2v_file = './input/models/GoogleNews-vectors-negative300.bin'
    tfidf_file = './input/models/sentences_dev.txt'
    df = task_to_df(task, dataset)
    return get_w2v_context_embeddings(df, w2v_file, tfidf_file=tfidf_file)

# ------------------------------------------------------
