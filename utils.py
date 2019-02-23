import pandas as pd
from pathlib import Path


def df_to_csv(df, path):
    df.to_csv(path, sep='\t', index=False, encoding='utf-8')


def csv_to_df(path):
    df = pd.read_csv(path, sep='\t', dtype=str, encoding='utf-8')
    return df


def max_arguments(task):
    fp = open(task, 'r')
    lines_args = fp.readlines()
    maxT = 0
    for line in lines_args:
        tokens = len(line.split(' '))
        if tokens > maxT:
            maxT = tokens
    return maxT - 3  # context_id, verb pos, verb-frame


def max_frameArguments(dataset):
    dir = "./semeval_data"
    task21_auto = dir + "/dev/auto/task-2.1.auto.txt"
    task21_dev = dir + "/dev/task-2.1.txt"
    task21_test =dir+"/test/task-2.1.txt"



    if dataset == 'dev':
        task21 = task21_dev
    elif dataset == 'auto':
        task21 = task21_auto
    elif dataset == 'test':
        task21 = task21_test


    return max_arguments(task21)
# ------------------------------------------------------------- df input from txt
import ud2csv

dir = "./semeval_data"
ud_gold = dir+"/dep-stx/pos-gold-dep-auto.conll.txt"
# -----------------------------------
def task_to_df(task, dataset):

    if Path('./input/train_task{}_{}.csv'.format(task, dataset)).exists():
        return csv_to_df('./input/train_task{}_{}.csv'.format(task, dataset))
    else:
        if task==1:
            return ud2csv.task1_to_df(dir+'/{}/task-1.txt'.format(dataset), ud_gold)
        if task ==22:
            return ud2csv.task22_to_df(dir + '/{}/task-2.2.txt'.format(dataset), ud_gold)


def task1_to_df_gd(dataset):
    if Path('./input/train_task{}_{}.csv'.format(1, dataset)).exists():
        return csv_to_df('./input/gd_task{}_{}.csv'.format(1, dataset))

    else:
        return ud2csv.task1_to_df_gd(dir+'/{}/task-1.txt'.format(dataset), ud_gold)


def task22_baselines(dataset, gr='in'):

    if Path('./input/all_grammaticalLabels_{}.csv'.format(dataset)).exists():
        df_task22 = csv_to_df('./input/all_grammaticalLabels_{}.csv'.format(dataset))
    else:
        df_task22 = ud2csv.task22_to_df_withFrameArgsDependencies(dir+'/{}/task-2.2.txt'.format(dataset), ud_gold)
    return ud2csv.getGrammaticalBaseline(df_task22, gr)
