import numpy as np
from elmo import load_elmo_context_embeddings
from word_embeddings import get_w2v_word_embeddings, get_elmo_word_embeddings, get_w2v_multiword_embeddings
from word2vec import get_w2v_context_embeddings_Default
from utils import task_to_df
import pathlib
# ---------------------------------------
def write_vectors():
    dir = 'vectors'

    tasks = ['1', '22']
    datasets = ['dev', 'test']

    for task in tasks:
        for dataset in datasets:

            pathlib.Path('{}/{}'.format(dir, dataset)).mkdir(parents=True, exist_ok=True)

            print(dataset,'task',task)
            df = task_to_df(task, dataset)

            w2v_context = get_w2v_context_embeddings_Default(task, dataset)
            outfile = '{}/{}/task{}_{}.npy'.format(dir, dataset, task,'w2v_context')
            np.save(outfile, w2v_context)

            elmo_word = get_elmo_word_embeddings(df['word'])
            outfile = '{}/{}/task{}_{}.npy'.format(dir, dataset, task, 'elmo_word')
            np.save(outfile, elmo_word)

            elmo_context = load_elmo_context_embeddings(task, dataset)
            outfile = '{}/{}/task{}_{}.npy'.format(dir, dataset, task, 'elmo_context')
            np.save(outfile, elmo_context)

            if task == '1':
                w2v_word = get_w2v_word_embeddings(df['word'])
                outfile = '{}/{}/task{}_{}.npy'.format(dir, dataset, task, 'w2v_word')
                np.save(outfile, w2v_word)

            else:
                w2v_word = get_w2v_multiword_embeddings(df['word'])
                outfile = '{}/{}/task{}_{}.npy'.format(dir, dataset, task, 'w2v_word')
                np.save(outfile, w2v_word)

                w2v_verb = get_elmo_word_embeddings(df['verb'])
                outfile = '{}/{}/task{}_{}.npy'.format(dir, dataset, task, 'w2v_verb')
                np.save(outfile, w2v_verb)

                elmo_verb = get_elmo_word_embeddings(df['verb'])
                outfile = '{}/{}/task{}_{}.npy'.format(dir, dataset, task, 'elmo_verb')
                np.save(outfile, elmo_verb)
# ---------------------------------------
import ud2csv
from utils import df_to_csv, csv_to_df
def write_input_CSVs():
    dir = "./semeval_data"

    ud_test = dir+"/dep-stx/pos-gold-dep-auto.conll.txt"
    task1_test = dir+"/test/task-1.txt"
    # task21_test = dir+"/test/task-2.1.txt"
    task22_test = dir+"/test/task-2.2.txt"
    #-----------------------------------------------
    ud_dev = dir+"/dep-stx/pos-gold-dep-auto.conll.txt"
    task1_dev = dir+"/dev/task-1.txt"
    # task21_dev = dir+"/dev/task-2.1.txt"
    task22_dev = dir+"/dev/task-2.2.txt"

    all_sentences_dev = './input/models/sentences_dev.txt'
    print('writing sentences for dev')
    ud2csv.ud_sentences_to_file(ud_dev, all_sentences_dev)

    print('writing csvs for dev')

    csv_dev = './input/train_task1_dev.csv'
    csv_gd_dev = './input/gd_task1_dev.csv'

    ud2csv.task1_to_csv(task1_dev, ud_dev, csv_dev)
    ud2csv.task1_to_csv_gd(task1_dev, ud_dev, csv_gd_dev)

    csv_task22_dev = './input/train_task22_dev.csv'
    ud2csv.task22_to_csv(task22_dev, ud_dev, csv_task22_dev)

    csv_gr_dev = './input/all_grammaticalLabels_dev.csv'
    df_task22 = ud2csv.task22_to_df_withFrameArgsDependencies(task22_dev, ud_dev)
    df_to_csv(df_task22, csv_gr_dev)
    # ------------------------------------------------------------- Test
    print('writing csvs for test')

    csv_test = './input/train_task1_test.csv'
    csv_gd_test = './input/gd_task1_test.csv'

    ud2csv.task1_to_csv(task1_test, ud_test, csv_test)
    ud2csv.task1_to_csv_gd(task1_test, ud_test, csv_gd_test)


    csv_task22_test = './input/train_task22_test.csv'
    ud2csv.task22_to_csv(task22_test, ud_test, csv_task22_test)

    csv_gr_test = './input/all_grammaticalLabels_test.csv'
    df_task22 = ud2csv.task22_to_df_withFrameArgsDependencies(task22_test, ud_test)
    df_to_csv(df_task22, csv_gr_test)

# ---------------------------------------
# dump elmo default layer for contexts, google word2vec, tfidf matrix
from elmo import dump_elmo_context_embeddings
from word2vec import load_w2v_model, load_tfidf

def dump_models_resources(w2v_file='./input/models/GoogleNews-vectors-negative300.bin'):

    tasks = ['1', '22']
    datasets = ['dev', 'test']
    for task in tasks:
        for dataset in datasets:
            dump_elmo_context_embeddings(task, dataset)

    w2v = load_w2v_model(w2v_file, normalized_w2v=False)
    tfidf_file = './input/models/sentences_dev.txt'
    load_tfidf(tfidf_file, w2v.index2word)


# ------------------------------------------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-w2v', '--w2v_file', help="path to GoogleNews-vectors-negative300.bin?",
                    default='./input/models/GoogleNews-vectors-negative300.bin')
args = parser.parse_args()

write_input_CSVs()
write_vectors()
dump_models_resources(args.w2v_file)