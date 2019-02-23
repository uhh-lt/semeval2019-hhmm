from utils import csv_to_df, df_to_csv
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
import sklearn
from sklearn.linear_model import LogisticRegression
from vectorizer import vectorize
#--------------------------------------------------------------------------- Task-1
def task1_toTXT(goldfile, df, output_dir):

    with open(goldfile, 'r') as fp:
        lines = fp.readlines()

    for i in range(len(lines)):
        line = lines[i]
        tokens = line.split(' ')
        cid, verb_index, verb = tokens[0], tokens[1], tokens[2].split('.')[0]
        subdf = df[(df['context_id'] == cid) & (df['word_index'] == verb_index) & (df['word'] == verb)]

        word, prediction = list(subdf['word'])[0], list(subdf['predict_sense_id'])[0]

        newline = '{} {} {}'.format(tokens[0], tokens[1], word + '.' + str(prediction))  # context_id, verbpos, verb.na
        lines[i] = newline + ' '

    outfile = '{}/{}'.format(output_dir, 'task-1.txt')
    with open(outfile, 'w') as fp:
        for line in lines:
            fp.write('{}\n'.format(line))

## -----------------------------------------------
from utils import task_to_df

def task1Submission(parameters, write=False, output_dir=None):
    print('\nTask-1 submission----------\n')
    print(parameters)

    dataset = 'test'
    df_out = task_to_df(1, dataset)
    txtfile = 'semeval_data/{}/task-1.txt'.format(dataset)

    method, norm, affinity, linkage, n_cluster = parameters

    clusterizer = AgglomerativeClustering(n_clusters=n_cluster, affinity=affinity, linkage=linkage)
    vecs = vectorize(method, task=1, dataset='test')

    if norm == True:
        method = method + '-norm'
        vecs = sklearn.preprocessing.normalize(vecs, norm='l2', axis=1)

    df_out['predict_sense_id'] = clusterizer.fit_predict(vecs)

    if write == True:
        # csvfile = '{}/task1_{}_#{}#{}#{}.csv'.format(output_dir, method, affinity, linkage, n_cluster)
        # df_to_csv(df_out, csvfile)
        # print(method + ' csv file is written')
        task1_toTXT(txtfile, df_out, output_dir)
        print('Task-1 txt file is written')

    return df_out


#--------------------------------------------------------------------------- Task-22
def task22_toTXT(goldfile, df, output_dir):
    df['context_id2'] = df['context_id'].apply(lambda x: x.split('_')[0])
    with open(goldfile, 'r') as fp:
        lines = fp.readlines()

    for i in range(len(lines)):

        line = lines[i]
        tokens = line.split(' ')
        cid, verb_index, verb = tokens[0], tokens[1], tokens[2].split('.')[0]
        subdf = df[(df['context_id2'] == cid) & (df['verb_index'] == verb_index)]
        ids, words, predictions = list(subdf['context_id']), list(subdf['word']), list(subdf['predict_sense_id'])

        newline = '{} {} {}'.format(tokens[0], tokens[1], tokens[2])  # context_id, verbpos, verb.na
        if len(tokens) > 3:
            tokens = tokens[3:]  # roles
            for token, idd, word, prediction in zip(tokens, ids, words, predictions):
                wpr = token.split('-:-')
                newline = newline + ' ' + '{}-:-{}-:-{}'.format(wpr[0], wpr[1], prediction)

        newline = newline.replace('\n', '')  # problem with #20436016

        lines[i] = newline + ' '

    outfile = '{}/{}'.format(output_dir, 'task-2.2.txt')
    with open(outfile, 'w') as fp:
        for line in lines:
            fp.write('{}\n'.format(line))


##----------------------------
def task22Submission_AC(parameters, write=False, output_dir=None):
    print(parameters)

    dataset = 'test'
    df_out = task_to_df(22, dataset)
    txtfile = 'semeval_data/{}/task-2.2.txt'.format(dataset)

    method, norm, affinity, linkage, n_cluster = parameters
    clusterizer = AgglomerativeClustering(n_clusters=n_cluster, affinity=affinity, linkage=linkage)
    vecs = vectorize(method, task=22, dataset='test')

    if norm == True:
        method = method + '-norm'
        vecs = sklearn.preprocessing.normalize(vecs, norm='l2', axis=1)

    df_out['predict_sense_id'] = clusterizer.fit_predict(vecs)
    # ---------------------------------------
    if write == True:
        # csvfile = '{}/task22_{}_#{}#{}#{}.csv'.format(output_dir, method, affinity, linkage, n_cluster)
        # df_to_csv(df_out, csvfile)
        # print(method + ' csv file is written')
        task22_toTXT(txtfile, df_out, output_dir)
        print('Task22 txt file is written')
    # ---------------------------------------
    return df_out


##----------------------------
def task22Submission_LR(parameters, write=False, output_dir=None):
    print(parameters)

    dataset = 'test'
    df_out = task_to_df(22, dataset)
    txtfile = 'semeval_data/{}/task-2.2.txt'.format(dataset)
    method, _ = parameters
    train_df = task_to_df(22, 'dev')
    train_y = train_df['gold_sense_id']
    train_X = vectorize(method, task=22, dataset='dev')
    test_X = vectorize(method, task=22, dataset='test')

    clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
    clf.fit(train_X, train_y)
    T = clf.predict_proba(test_X)
    confidence = np.amax(T, axis=1)

    y_pred = clf.predict(test_X)
    df_out['predict_sense_id'] = y_pred
    # ---------------------------------------
    if write == True:
        # csvfile = '{}/task22_{}_#{}.csv'.format(output_dir, method, ('LogisticRegression'))
        # df_to_csv(test_df, csvfile)
        # print(method + ' csv file is written')
        task22_toTXT(txtfile, df_out, output_dir)
        print('Task-22 txt file is written')
    # ---------------------------------------
    return df_out, confidence

# ---------------------------------------
def task22Submission_joint(parameters, threshold='concat', write=False, output_dir=None):

    dataset = 'test'
    txtfile = 'semeval_data/{}/task-2.2.txt'.format(dataset)

    method, norm, affinity, linkage, n_cluster = parameters
    df_out, confidence = task22Submission_LR((method, 'LR'), write=False)
    print('computed supervised labels for {}'.format(method))

    df_out2 = task22Submission_AC((method, 'False', 'cosine', 'average', 34), write=False)

    df_out['supervised_labels'] = df_out['predict_sense_id']
    df_out['clustering_labels'] = df_out2['predict_sense_id']

    joint_predict_sense_id = []
    if threshold == 'concat':
        joint_predict_sense_id = df_out['supervised_labels'].astype(str) + '_' + df_out['clustering_labels'].astype(str)
    else:
        for c, supervised_label, clustering_label in zip(confidence, df_out['supervised_labels'], df_out['clustering_labels']):
            if c >= threshold:
                joint_predict_sense_id.append(supervised_label)
            else:
                joint_predict_sense_id.append(str(clustering_label))

    df_out['predict_sense_id'] = joint_predict_sense_id

    # ---------------------------------------
    if write == True:
        # csvfile = '{}/task22_{}_#{}_{}.csv'.format(output_dir, method, 'JointClustering', threshold)
        # df_to_csv(df_out, csvfile)
        # print(method + ' csv file is written')
        task22_toTXT(txtfile, df_out, output_dir)
        print('Task-22  txt file is written')
    # ---------------------------------------
    return df_out

# ---------------------------------------
def task22Submission(parameters, write=False, output_dir=None):
    print('\nTask-22 submission----------\n')
    print(parameters)

    if 'LR' in parameters:
        task22Submission_LR(parameters, write, output_dir)

    elif 'joint' in parameters:
        task22Submission_joint(parameters[0:5], parameters[-1],  write, output_dir)

    else:
        task22Submission_AC(parameters, write, output_dir)

#--------------------------------------------------------------------------- Task-2.1
from collections import namedtuple, OrderedDict
def merge(task1f, task21f, task22f, outputf, placeholder='unknown'):
    print('\nTask-21 submission----------\n')

    task1 =  open(task1f, 'r')
    task21 =  open(task21f, 'r')
    task22 =  open(task22f, 'r')
    output =  open(outputf, 'w')

    Frame = namedtuple('Frame', 'id positions verb slots')
    Slot = namedtuple('Slot', 'word positions role')

    frames = OrderedDict()
    for task21_lines, line in enumerate(task21):
        row = line.rstrip().split(' ')
        # print(line)
        frame_id, positions, _ = row[0], row[1], row[2]

        slots = OrderedDict()

        for record in row[3:]:
            word, word_positions, _ = record.split('-:-', 2)

            slot_id = word + '_' + word_positions
            slots[slot_id] = Slot(word, word_positions, None)

        id = frame_id + '_' + positions
        frames[id] = Frame(frame_id, positions, None, slots)

    for task1_lines, line in enumerate(task1):
        frame_id, positions, verb = line.rstrip().split(' ', 2)

        id = frame_id + '_' + positions
        assert id in frames, id

        frames[id] = frames[id]._replace(verb=verb)

    assert task21_lines == task1_lines, (task21_lines, task1_lines)
    assert all(frame.verb is not None for frame in frames.values()), 'empty verbs found'

    for task22_lines, line in enumerate(task22):
        row = line.rstrip().split(' ')

        frame_id, positions, _ = row[0], row[1], row[2]
        id = frame_id + '_' + positions

        for record in row[3:]:
            word, word_positions, role = record.split('-:-', 2)
            slot_id = word + '_' + word_positions

            if slot_id in frames[id].slots:
                frames[id].slots[slot_id] = frames[id].slots[slot_id]._replace(role=role)

    assert task21_lines == task22_lines, (task21_lines, task22_lines)

    for frame in frames.values():
        for slot_id, slot in frame.slots.items():
            if slot.role is None:
                if placeholder == 'boolean':
                    verb_position = int(frame.positions.split('_')[0])
                    word_position = int(slot.positions.split('_')[0])
                    role = 'Left' if word_position < verb_position else 'Right'
                else:
                    role = 'UKN'

                frame.slots[slot_id] = slot._replace(role=role)

    assert all(slot.role is not None for frame in frames.values() for slot in frame.slots.values()), 'empty roles still found'

    for frame in frames.values():
        slots = ' '.join(['-:-'.join(slot) for slot in frame.slots.values()])
        print(' '.join((frame.id, frame.positions, frame.verb, slots)), file=output)

# -----------------------------------
def task21Submission(output_dir):
    merge(output_dir + '/task-1.txt', 'semeval_data/test/task-2.1.txt', output_dir + '/task-2.2.txt',
                     output_dir + '/task-2.1.txt')

    print(' -- Task-21  txt file is written')

# ----------------------------------------------------------------------
