# =============================================================================
from pathlib import Path
import numpy as np
from elmo import load_elmo_context_embeddings
from word_embeddings import get_w2v_word_embeddings, get_elmo_word_embeddings, get_w2v_multiword_embeddings
from word_embeddings import get_w2v_so_embeddings
from word2vec import get_w2v_context_embeddings_Default


from utils import task_to_df, task22_baselines
from ud2csv import getLabelNames

def vectorize_baseline(dataset, baselines='inbound_deps'):
    datset_for_labels = 'dev'
    df = task22_baselines(dataset, 'in')

    grLabels = []
    for baseline in baselines.split('+'):
        if baseline =='bool_pos':
            grLabels.append('boolean_position_to_verb')

        elif baseline == '123_pos':
            grLabels.append('position_to_verb')

        elif baseline == 'inbound_deps':
            grLabels.extend(getLabelNames(task22_baselines(datset_for_labels, 'in'), ['inbound_dependency']))

        else:
            df = task22_baselines(dataset, baseline)
            grLabels.extend(getLabelNames(task22_baselines(datset_for_labels, 'inout'), ['inbound_dependency', 'outbound_dependency']))


    vectors = np.array([])
    for label in grLabels:
        if label in df.columns:
            v = np.array(list(df[label])).reshape(len(df), 1)
            v = v.astype(int)
        else:
            v = np.zeros(len(df)).reshape(len(df), 1)

        if vectors.size == 0:
            vectors = v
        else:
            vectors = np.hstack((vectors, v))

    return vectors

# ----------------------------------------------------------------------------------
def context_embeddings(task, dataset, model, emb='context'):

    dir = 'vectors'
    file = '{}/{}/task{}_{}_{}.npy'.format(dir, dataset, task, model, emb)
    if Path(file).exists():
        return np.load(file)
    else:
        if model=='w2v':
            return get_w2v_context_embeddings_Default(task, dataset)
        else:
            return load_elmo_context_embeddings(task, dataset)


def word_embeddings(task, dataset, model, emb='word'):

    dir = 'vectors'
    file = '{}/{}/task{}_{}_{}.npy'.format(dir, dataset, task, model, emb)
    if Path(file).exists():
        return np.load(file)
    else:
        df = task_to_df(task, dataset)

        if model == 'w2v':
            if task ==1:
                return get_w2v_word_embeddings(df['word'])
            else:
                return word_embeddings(task, dataset, model='w2v')

        else:
            return get_elmo_word_embeddings(df['word'])


def verb_embeddings(task, dataset, model, emb='verb'):

    dir = 'vectors'
    file = '{}/{}/task{}_{}_{}.npy'.format(dir, dataset, task, model, emb)
    if Path(file).exists():
        return np.load(file)
    else:
        df = task_to_df(task, dataset)
        if model=='w2v':
            return get_w2v_word_embeddings(df['verb'])
        else:
            return get_elmo_word_embeddings(df['verb'])


# ----------------------------------------------------------------------------------
def vectorize(embedding_name, task, dataset):
    dir = './vectors'
    file = '{}/{}/task{}_{}.npy'.format(dir, dataset, task, embedding_name)
    if Path(file).exists():
        vecs = np.load(file)
    else:
        #---------------------------------------------------------------------------------- Task-1
            if embedding_name == 'w2v_word':
                vecs = word_embeddings(task, dataset, model='w2v')

            if embedding_name == 'w2v_context+w2v_word':
                vecs = np.hstack((context_embeddings(task, dataset, model='w2v'),
                                  word_embeddings(task, dataset, model='w2v')))

            if embedding_name == 'elmo_context+elmo_word':
                vecs = np.hstack((context_embeddings(task, dataset, model='elmo'),
                                  word_embeddings(task, dataset, model='elmo')))

            if embedding_name == 'w2v_context+w2v_word+elmo_context+elmo_word':
                vecs = np.hstack((context_embeddings(task, dataset, model='w2v'),
                                  word_embeddings(task, dataset, model='w2v'),
                                  context_embeddings(task, dataset, model='elmo'),
                                  word_embeddings(task, dataset, model='elmo')))

            if embedding_name == 'w2v_context+w2v_word+w2v_so':
                vecs = np.hstack((context_embeddings(task, dataset, model='w2v'),
                                  word_embeddings(task, dataset, model='w2v'),
                                  get_w2v_so_embeddings(dataset, 'concat')))

        # ---------------------------------------------------------------------------------- Task-2.2

            if embedding_name == 'w2v_context+inbound_deps':
                vecs = np.hstack((context_embeddings(task, dataset, model='w2v'),
                                  vectorize_baseline(dataset, baselines='inbound_deps')))

            if embedding_name == 'inbound_deps':
                vecs = vectorize_baseline(dataset, baselines='inbound_deps')

            if embedding_name == 'w2v_word+inbound&outbound':
                vecs = np.hstack((word_embeddings(task, dataset, model='w2v'),
                                  vectorize_baseline(dataset, baselines='inbound_deps')))

            if embedding_name == 'elmo_context+elmo_word':
                vecs = np.hstack((context_embeddings(task, dataset, model='elmo'),
                                  word_embeddings(task, dataset, model='elmo')))

            if embedding_name == 'elmo_context+elmo_word+inbound_deps+bool_pos':
                vecs = np.hstack((context_embeddings(task, dataset, model='elmo'),
                                  word_embeddings(task, dataset, model='elmo'),
                                  vectorize_baseline(dataset, baselines='inbound_deps+bool_pos')))

            if embedding_name == 'elmo_word+inbound_deps+bool_pos':
                vecs = np.hstack((word_embeddings(task, dataset, model='elmo'),
                                  vectorize_baseline(dataset, baselines='inbound_deps+bool_pos')))

            if embedding_name == 'w2v_context+bool_pos+123_pos':
                vecs = np.hstack((context_embeddings(task, dataset, model='w2v'),
                                  vectorize_baseline(dataset, baselines='bool_pos+123_pos')))

            if embedding_name == 'elmo_context+elmo_word+inbound_deps+bool_pos+123_pos':
                vecs = np.hstack((context_embeddings(task, dataset, model='elmo'),
                                  word_embeddings(task, dataset, model='elmo'),
                                  vectorize_baseline(dataset, baselines='inbound_deps+bool_pos+123_pos')))

            if embedding_name == 'elmo_word+inbound_deps+bool_pos+123_pos':
                vecs = np.hstack((word_embeddings(task, dataset, model='elmo'),
                                  vectorize_baseline(dataset, baselines='inbound_deps+bool_pos+123_pos')))

            if embedding_name == 'elmo_context+elmo_word+elmo_verb+inbound_deps+bool_pos+123_pos':
                vecs = np.hstack((context_embeddings(task, dataset, model='elmo'),
                                  word_embeddings(task, dataset, model='elmo'),
                                  verb_embeddings(task, dataset, model='elmo'),
                                  vectorize_baseline(dataset, baselines='inbound_deps+bool_pos+123_pos')))

            if embedding_name == 'elmo_context+inbound_deps':
                vecs = np.hstack((context_embeddings(task, dataset, model='elmo'),
                                  vectorize_baseline(dataset, baselines='inbound_deps')))

            if embedding_name == 'w2v_context+w2v_word+inbound_deps+bool_pos':
                vecs = np.hstack((context_embeddings(task, dataset, model='w2v'),
                                  word_embeddings(task, dataset, model='w2v'),
                                  vectorize_baseline(dataset, baselines='inbound_deps+bool_pos')))

            if embedding_name == 'w2v_word+inbound_deps+bool_pos':
                vecs = np.hstack((word_embeddings(task, dataset, model='w2v'),
                                  vectorize_baseline(dataset, baselines='inbound_deps+bool_pos')))

            if embedding_name == 'w2v_context+w2v_word+inbound_deps+bool_pos+123_pos':
                vecs = np.hstack((context_embeddings(task, dataset, model='w2v'),
                                  word_embeddings(task, dataset, model='w2v'),
                                  vectorize_baseline(dataset, baselines='inbound_deps+bool_pos+123_pos')))

            if embedding_name == 'w2v_word+inbound_deps+bool_pos+123_pos':
                vecs = np.hstack((word_embeddings(task, dataset, model='w2v'),
                                  vectorize_baseline(dataset, baselines='inbound_deps+bool_pos+123_pos')))

            if embedding_name == 'w2v_context+w2v_word+w2v_verb+inbound_deps+bool_pos+123_pos':
                vecs = np.hstack((context_embeddings(task, dataset, model='w2v'),
                                  word_embeddings(task, dataset, model='w2v'),
                                  verb_embeddings(task, dataset, model='w2v'),
                                  vectorize_baseline(dataset, baselines='inbound_deps+bool_pos+123_pos')))

    return vecs
# -------------------------------------------------------
