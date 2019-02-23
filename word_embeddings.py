# ---------------------------------------------
from word2vec import load_w2v_model
import pandas as pd
import numpy as np
from pathlib import Path

from elmo import load_elmo_context_embeddings
from utils import df_to_csv, csv_to_df, max_frameArguments, max_arguments
from utils import task_to_df, task1_to_df_gd

# ---------------------------------------------------- W2V
def get_w2v_wordAvg(tokens, w2v):
    v = np.zeros(300).reshape(1, 300)
    for w in tokens:
        if not w in w2v.index2word:
            #             print('not found in w2v--'+w)
            wv = np.zeros(300).reshape(1, 300)
        else:
            wv = w2v.get_vector(w).reshape(1, 300)

        v = v + wv  # sum
    # to average out the multiword expression
    v = v / len(tokens)
    return v


def get_w2v_multiword_embeddings(words, w2v=None, normalized_w2v=False):
    if w2v == None:
        w2v = load_w2v_model(None, normalized_w2v)

    vecs = np.array([])
    for word in words:
        if type(word) == float and np.isnan(word) == True:
            v = np.zeros(300).reshape(1, 300)
        else:
            if not word in w2v.index2word:
                #                 print('not found in w2v--'+word)
                tokens = word.split('_')
                v = get_w2v_wordAvg(tokens, w2v)
            else:
                v = w2v.get_vector(word).reshape(1, 300)

        if vecs.size == 0:
            vecs = v
        else:
            vecs = np.vstack((vecs, v))
    return vecs


def get_w2v_word_embeddings(words, normalized_w2v=False):
    w2v = load_w2v_model(None, normalized_w2v)
    word_embeddings = np.array([])
    for word in words:

        if type(word) == float and np.isnan(word) == True:
            v = np.zeros(300).reshape(1, 300)
        else:
            w = word.split('_')[0]  # in semeval dataset _ is used as connector for multi-words verb
            if not w in w2v.index2word:
                v = np.zeros(300).reshape(1, 300)
            else:
                v = w2v.get_vector(w).reshape(1, 300)

        if word_embeddings.size == 0:
            word_embeddings = v
        else:
            word_embeddings = np.vstack((word_embeddings, v))
    return word_embeddings


def get_w2v_so_embeddings(dataset, op='concat', normalized_w2v=False):
    dataset_file = './input/gd_{}.csv'.format(dataset)

    if not Path(dataset_file).exists():
        df = task1_to_df_gd(dataset)
    else:
        df = csv_to_df(dataset_file)

    subjects, objects = df['subject_lemma'], df['object_lemma']

    w2v = load_w2v_model(None, normalized_w2v)
    embeddings = np.array([])

    for s, o in zip(subjects, objects):

        non_zeros = 0

        if type(s) == float and np.isnan(s) == True:
            sv = np.zeros(300).reshape(1, 300)
        else:
            non_zeros = non_zeros + 1

            w = s.split('_')[0]
            if not w in w2v.index2word:
                sv = np.zeros(300).reshape(1, 300)
            else:
                sv = w2v.get_vector(w).reshape(1, 300)

        if type(o) == float and np.isnan(o) == True:
            ov = np.zeros(300).reshape(1, 300)
        else:
            non_zeros = non_zeros + 1

            w = o.split('_')[0]
            if not w in w2v.index2word:
                ov = np.zeros(300).reshape(1, 300)
            else:
                ov = w2v.get_vector(w).reshape(1, 300)

        if op == 'concat':
            # svo concatenate
            so_v = np.hstack((sv, ov))
        elif op == 'sum':
            so_v == sv + ov
        else:  # avg
            so_v == (sv + ov) / non_zeros

        if embeddings.size == 0:
            embeddings = so_v
        else:
            embeddings = np.vstack((embeddings, so_v))

    print(embeddings.shape)
    return embeddings


# ---------------------------------------------------- Elmo
def get_elmo_word_embeddings(words):
    words = list(words)
    # elmo help=https://github.com/PrashantRanjan09/Elmo-Tutorial/blob/master/Elmo_tutorial.ipynb
    embeddings_elmo = []
    import tensorflow as tf
    import tensorflow_hub as hub
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        embeddings_elmo = session.run(elmo(words,
                                           signature="default",
                                           as_dict=True)["default"])

    return embeddings_elmo


# there may be multi words as oppose to verbs, like buy_out, buy_back
def get_elmo_verb(df, i, context_embedding):
    word = df['word'][i]
    v_indicies = df['verb_index'][i].split('_')
    verb = df['verb_lemma'][i]
    embed = context_embedding[int(v_indicies[0])]

    if word != verb:
        for i in v_indicies[1:]:
            embed += context_embedding[int(i)]
    return embed
