import logging
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from  pathlib import Path
from utils import task_to_df


# ----------------------------------------------------------
def get_elmo_context_embeddings(task, dataset, ly= 'default'):

    df = task_to_df(task, dataset)

    sentences = list(df['context'])

    N = len(df)
    print('Total Records--{}'.format(N))

    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

    BATCH=1000
    embeddings_elmo=np.array([])
    for i in range(0, N, BATCH):
        if ly == 'default':
            tensors=elmo(sentences[i:i+BATCH], signature = "default", as_dict = True)[ly]

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            embeddings = session.run(tensors)

        print('{}-{} records processed...'.format(i, i+BATCH))

        if i==0:
            embeddings_elmo = embeddings
        else:
            embeddings_elmo = np.vstack((embeddings_elmo, embeddings))

    logging.info('{} layer shape--{}'.format(ly, embeddings_elmo.shape))
    return embeddings_elmo

# ----------------------------------------------------------
def dump_elmo_context_embeddings(task, dataset, ly='default'):
    output_dir = './input/models/'
    dataset_file = './input/train_task{}_{}.csv'.format(task, dataset)
    dump = '{}elmo_tf_{}_task{}_{}.pkl'.format(output_dir, ly, task, dataset)
    vecs = get_elmo_context_embeddings(task, dataset, ly)
    f = open(dump, 'wb')
    pickle.dump(vecs, f, -1)
    f.close()


def load_elmo_context_embeddings(task, dataset, ly='default'):
    output_dir = './input/models/'
    dump = '{}elmo_tf_{}_task{}_{}.pkl'.format(output_dir, ly, task, dataset)
    if Path().exists():
        f = open(dump, 'rb')
        vecs = pickle.load(f)
        f.close()
        return vecs
    else:
        return get_elmo_context_embeddings(task, dataset, ly)

