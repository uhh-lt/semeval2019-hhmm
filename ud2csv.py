# http://universaldependencies.org/format.html

import pyconll
import pandas as pd
from utils import df_to_csv, csv_to_df, max_arguments


def ud_sentence_to_dict(udfile):  # just put space between each word token of tree, without any other processing

    data = pyconll.load_from_file(udfile)
    context_dict = {}

    for sentence in data:
        s_id = sentence.source.splitlines()[0]  # sentence id
        _sentence = ""
        #       print(_id)
        for token in sentence:

            wlist = token.conll().split()
            w = wlist[1]  # word
            # --------------- Replace 
            if w == '-LRB-':
                w = '('
            if w == '-RRB-':
                w = ')'
            # ----------------

            if not _sentence:
                _sentence = w
                continue

            else:
                _sentence = _sentence + ' ' + w

        # print(_sentence)
        context_dict[s_id] = _sentence
    return context_dict


def ud_to_dict(udfile):
    data = pyconll.load_from_file(udfile)

    tree_dict = {}
    for sentence in data:
        sent_lines = sentence.source.splitlines()
        s_id = sent_lines[0]
        tree_dict[s_id] = sent_lines[1:]

    return tree_dict


def dict_to_df(context_dict):
    # open the txt files
    labels = ['context_id', 'context']
    df = pd.DataFrame(columns=labels)
    context = []
    context_id = []
    for key, value in context_dict.items():
        context_id.append(key)
        context.append(value)
    df['context_id'] = context_id
    df['context'] = context
    return df


def task1_to_df(annotated_file, udfile):
    context_dict = ud_sentence_to_dict(udfile)
    # open the txt files
    fp = open(annotated_file, 'r')
    lines = fp.readlines()

    labels = ['context_id', 'word', 'word_index', 'gold_sense_id', 'predict_sense_id', 'context']
    df = pd.DataFrame(columns=labels)

    for i in range(len(lines)):
        frame_tokens = lines[i].split()
        sentence = context_dict.get(frame_tokens[0])  # frame_tokens[0] is conext_id
        word, sense = frame_tokens[2].split('.')
        df.loc[i] = [frame_tokens[0], word, frame_tokens[1], sense, '', sentence]

    return df


def task22_to_df(task22_file, udfile):
    context_dict = ud_sentence_to_dict(udfile)
    fp = open(task22_file, 'r')
    lines_args = fp.readlines()
    labels = ['context_id', 'word', 'gold_sense_id', 'predict_sense_id', 'context',
              'verb', 'verb_index', 'role_index']

    df = pd.DataFrame(columns=labels)

    for i in range(len(lines_args)):

        frame_tokens = lines_args[i].split(' ')
        s_id = frame_tokens[0]
        sentence = context_dict[s_id]
        # example--> #20064023 2 begin.na Ford-:-1-:-Agent install-:-3-:-Theme
        verb_index, verb = frame_tokens[1], frame_tokens[2].split('.')[0]
        roles = frame_tokens[3:]  # 0 is s_id, 1 is verb_position and 2 is verb.Frame
        s = 0
        for role in roles:  # if a line does not have any role, it will be skipped

            role_lemma, role_index, gold_label = role.replace('\n', '').split('-:-')
            row = ['{}_{}'.format(s_id, s), role_lemma, gold_label, '', sentence, verb, verb_index, role_index]
            df.loc[len(df)] = row
            s = s + 1
    return df


def task1_to_df_svo(task1_file, udfile):
    # open the txt files
    tree_dict = ud_to_dict(udfile)
    context_dict = ud_sentence_to_dict(udfile)

    fp = open(task1_file, 'r')
    lines = fp.readlines()

    labels = ['context_id', 'word', 'word_index', 'gold_sense_id', 'predict_sense_id', 'context',
              'verb_lemma', 'subject_lemma', 'object_lemma',
              'verb_index', 'subject_index', 'object_index']

    df = pd.DataFrame(columns=labels)
    for i in range(len(lines)):

        frame_tokens = lines[i].split(' ')
        s_id = frame_tokens[0]
        sent_lines = tree_dict[s_id]

        sentence = context_dict[s_id]
        word_tokens = sentence.split(' ')

        vb_index = frame_tokens[1].split('_')[0]
        subj_index = 0
        obj_index = 0
        vb = ''
        subj = ''
        obj = ''
        for token in sent_lines:
            nodes = token.split()
            if nodes[0] == vb_index:
                vb = nodes[2]

            if (nodes[6] == vb_index):  # dependency at verb

                # subj
                if (nodes[7] == 'nsubj' or nodes[7] == 'nmod'):
                    subj_index = nodes[0]
                    subj = nodes[2]

                # obj
                if (nodes[7] == 'nsubjpass' or nodes[7] == 'dobj'):
                    obj_index = nodes[0]
                    obj = nodes[2]

        index = [int(vb_index), int(subj_index), int(obj_index)]
        lemmas = [vb, subj, obj]

        word, sense = frame_tokens[2].replace('\n', '').split('.')

        df.loc[i] = [s_id, word, frame_tokens[1], sense, '', sentence, lemmas[0], lemmas[1], lemmas[2], index[0],
                     index[1], index[2]]

    return df


def task1_to_df_gd(task1_file, udfile):
    # open the txt files
    tree_dict = ud_to_dict(udfile)
    context_dict = ud_sentence_to_dict(udfile)

    fp = open(task1_file, 'r')
    lines = fp.readlines()

    dependent_entries = []

    labels = ['context_id', 'word', 'word_index', 'gold_sense_id', 'predict_sense_id', 'context',
              'verb_lemma', 'subject_lemma', 'object_lemma', 'iobj_lemma', 'csubj_lemma', 'ccomp_lemma',
              'verb_index', 'subject_index', 'object_index', 'iobj_index', 'csubj_index', 'ccomp_index']
    df = pd.DataFrame(columns=labels)
    for i in range(len(lines)):

        frame_tokens = lines[i].split(' ')
        s_id = frame_tokens[0]
        sent_lines = tree_dict[s_id]

        sentence = context_dict[s_id]
        word_tokens = sentence.split(' ')

        vb_index = frame_tokens[1].split('_')[0]
        #         print(vb_index)

        subj_index = 0
        obj_index = 0
        iobj_index = 0
        csubj_index = 0
        ccomp_index = 0

        vb = ''
        subj = ''
        obj = ''
        iobj = ''
        csubj = ''
        ccomp = ''

        for token in sent_lines:
            nodes = token.split()
            if nodes[0] == vb_index:
                vb = nodes[2]

            if (nodes[6] == vb_index):

                # subj
                if (nodes[7] == 'nsubj' or nodes[7] == 'nmod'):
                    subj_index = nodes[0]
                    subj = nodes[2]

                # obj
                if (nodes[7] == 'nsubjpass' or nodes[7] == 'dobj'):
                    obj_index = nodes[0]
                    obj = nodes[2]

                if nodes[7] == 'iobj':
                    iobj_index = nodes[0]
                    iobj = nodes[2]
                    dependent_entries.append(nodes[7] + '-' + nodes[2])

                if nodes[7] == 'csubj':
                    csubj_index = nodes[0]
                    csubj = nodes[2]
                    dependent_entries.append(nodes[7] + '-' + nodes[2])

                if nodes[7] == 'ccomp':
                    ccomp_index = nodes[0]
                    ccomp = nodes[2]
                    dependent_entries.append(nodes[7] + '-' + nodes[2])

        vb_index = frame_tokens[1]  # in test data verb index is multi_integers

        index = [vb_index, int(subj_index), int(obj_index), int(iobj_index), int(csubj_index), int(ccomp_index)]
        lemma = [vb, subj, obj, iobj, csubj, ccomp]

        word, sense = frame_tokens[2].replace('\n', '').split('.')
        df.loc[i] = [s_id, word, frame_tokens[1], sense, '', sentence, lemma[0], lemma[1], lemma[2], lemma[3], lemma[4],
                     lemma[5], index[0], index[1], index[2], index[3], index[4], index[5]]

    return df  # , set(dependent_entries)


def task1_to_df_withFrameArgs(task21_file, udfile):
    context_dict = ud_sentence_to_dict(udfile)

    fp = open(task21_file, 'r')
    lines_args = fp.readlines()

    labels = ['context_id', 'word', 'word_index', 'gold_sense_id', 'predict_sense_id', 'context',
              'verb_lemma', 'verb_index'
              ]

    mr = max_arguments(task21_file)
    for n in range(1, mr + 1):
        labels.append('{}{}_{}'.format('arg', n, 'lemma'))
        labels.append('{}{}_{}'.format('arg', n, 'index'))

    print(labels)
    df = pd.DataFrame(columns=labels)

    for i in range(len(lines_args)):

        frame_tokens = lines_args[i].split(' ')
        s_id = frame_tokens[0]

        sentence = context_dict[s_id]
        word_tokens = sentence.split(' ')

        # example--> Ford-:-1-:-Agent install-:-3-:-Activity
        word, sense = frame_tokens[2].replace('\n', '').split('.')
        row = [s_id, word, frame_tokens[1], sense, '', sentence, word, frame_tokens[1]]

        roles = frame_tokens[3:]  # 0 is s_id, 1 is verb_position and 2 is verb.Frame

        n = 1
        for role in roles:
            role_lemma, role_index = role.split('-:-')[0:2]
            row.append(role_lemma)
            row.append(role_index)

            n = n + 1
        for j in range(n, mr + 1):
            row.append('')
            row.append('0')

        df.loc[i] = row

    return df

#-------------------------------------------------------------------------
def ud_sentences_to_file(udfile, target_file):
    di = ud_sentence_to_dict(udfile)
    df = dict_to_df(di)
    with open(target_file, 'w') as fp:
        for sent in df['context']:
            fp.write(sent + '\n')


def task1_to_csv(task1_file, udfile, target_file):
    df = task1_to_df(task1_file, udfile)
    df_to_csv(df, target_file)
    return df


def task22_to_csv(task22_file, udfile, target_file):
    df = task22_to_df(task22_file, udfile)
    df_to_csv(df, target_file)
    return df


def task1_to_csv_svo(task1_file, udfile, target_file):
    df = task1_to_df_svo(task1_file, udfile)
    df_to_csv(df, target_file)
    return df


def task1_to_csv_gd(task1_file, udfile, target_file):
    df = task1_to_df_gd(task1_file, udfile)
    df_to_csv(df, target_file)
    return df


def task1_sentences_to_file(task1_file, udfile, target_file):
    df = task1_to_df(task1_file, udfile)
    with open(target_file, 'w') as fp:
        for sent in df['context']:
            fp.write(sent + '\n')


#-------------------------------------------------------------------------
def back_to_verb(source, prevsource, sent_lines, verb_index):
    if source == '0':  # role is ROOT
        return sent_lines[int(prevsource)].split()[7]

    nextsource = sent_lines[int(source) - 1].split()[6]

    if nextsource == verb_index or nextsource == '0':
        #         print(nextsource, verb_index)
        if nextsource != verb_index:  # NOT RELATED TO TARGET VERB
            return sent_lines[int(prevsource) - 1].split()[7]
        else:
            return sent_lines[int(prevsource) - 1].split()[7]
    else:
        nextsource = sent_lines[int(source) - 1].split()[6]
        return back_to_verb(nextsource, source, sent_lines, verb_index)


def task22_to_df_withFrameArgsDependencies(task22_file, udfile):
    # open the txt files
    tree_dict = ud_to_dict(udfile)
    context_dict = ud_sentence_to_dict(udfile)

    fp = open(task22_file, 'r')
    lines_args = fp.readlines()

    labels = ['context_id', 'word', 'gold_sense_id', 'predict_sense_id', 'context',
              'verb', 'verb_index', 'role_index', 'inbound_dependency', 'outbound_dependency']#, 'dependency_to_verb']

    labels.append('position_to_verb')
    labels.append('boolean_position_to_verb')

    mr = max_arguments(task22_file)

    df = pd.DataFrame(columns=labels)

    for i in range(len(lines_args)):

        frame_tokens = lines_args[i].split(' ')
        s_id = frame_tokens[0]
        sent_lines = tree_dict[s_id]

        sentence = context_dict[s_id]
        word_tokens = sentence.split(' ')

        # example--> #20064023 2 begin.na Ford-:-1-:-Agent install-:-3-:-Theme
        verb_index, verb = frame_tokens[1], frame_tokens[2].split('.')[0]
        roles = frame_tokens[3:]  # 0 is s_id, 1 is verb_position and 2 is verb.Frame
        s = 0
        for role in roles:

            role_lemma, role_index, gold_label = role.replace('\n', '').split('-:-')
            row = ['{}_{}'.format(s_id, s), role_lemma, gold_label, '', sentence, verb, verb_index, role_index]
            inLabel = 'None'
            outLabel = 'None'
            backtoVerb = 'None'
            for index in role_index.split('_'):
                for token in sent_lines:
                    nodes = token.split()
                    # inbound grammatical dependency
                    if nodes[0] == index:
                        if inLabel == 'None':
                            inLabel = nodes[7]
                        else:
                            inLabel = inLabel + '_' + nodes[7]

                        # source = nodes[6]
                        # prevsource = nodes[0]

                    # outbound grammatical dependency
                    if nodes[6] == index:
                        if outLabel == 'None':
                            outLabel = nodes[7]
                        else:
                            outLabel = outLabel + '_' + nodes[7]

            # lastLabeltoVerb = back_to_verb(source, prevsource, sent_lines, verb_index.split('_')[0])

            indicies = role_index.split('_')
            pos = s + 1
            if int(indicies[0]) > int(verb_index.split('_')[0]):  # in test set format of index is changed
                bool_pos = 0
            else:
                bool_pos = 1

            row.append(inLabel)
            row.append(outLabel)
            # row.append(lastLabeltoVerb)

            row.append(pos)
            row.append(bool_pos)

            df.loc[len(df)] = row
            s = s + 1

    return df


def getLabelNames(df_task22, columns):
    labels = []
    for column in columns:
        for label in df_task22[column]:
            if label != 'None':
                for lbl in label.split('_'):
                    labels.append(lbl.split(':')[0])
    return list(set(labels))


def getGrammaticalBaseline(df_task22, gr='in'):
    #     df_task22=task22_to_df_withFrameArgsDependencies(task22_file, udfile)
    grLabels=[]
    if gr == 'in':
        grLabels = getLabelNames(df_task22, ['inbound_dependency'])

    if gr == 'out':
        grLabels = getLabelNames(df_task22, ['outbound_dependency'])

    if gr == 'inout':
        grLabels = getLabelNames(df_task22, ['inbound_dependency', 'outbound_dependency'])

    columns = list(df_task22.columns)
    for lb in grLabels:
        columns.append(lb)

    df = pd.DataFrame(columns=columns)
    # print(columns)

    for index, inLabel, outLabel in zip(df_task22.index, df_task22['inbound_dependency'],
                                        df_task22['outbound_dependency']):

        row = list(df_task22.loc[index])
        for lb in grLabels:
            row.append(0)

        if gr == 'in' or gr == 'inout':
            if inLabel != 'None':
                multiLabels = inLabel.split('_')
                for label in multiLabels:
                    column = label.split(':')[0]
                    row[columns.index(column)] = -1

        if gr == 'out' or gr == 'inout':
            if outLabel != 'None':
                multiLabels = outLabel.split('_')
                for label in multiLabels:
                    column = label.split(':')[0]
                    if row[columns.index(column)] == -1:  # also appear in inbound_dependency
                        row[columns.index(column)] = -1
                    else:
                        row[columns.index(column)] = 1

        df.loc[len(df)] = row

    return df

#-------------------------------------------------------------------------
