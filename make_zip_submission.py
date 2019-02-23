#!/usr/bin/python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# ----------------------------------------------------------------------------
submissions_task1 = {
    1: ('w2v_context+w2v_word', True, 'manhattan', 'average', 175),
    2: ('w2v_context+w2v_word', False, 'cosine', 'average', 175),
    3: ('elmo_context+elmo_word', False, 'cosine', 'average', 175),
    4: ('w2v_context+w2v_word', True, 'manhattan', 'average', 150),
    5: ('w2v_context+w2v_word+elmo_context+elmo_word', True, 'manhattan', 'average', 150),
    6: ('w2v_context+w2v_word+w2v_so', True, 'manhattan', 'average', 150),
    7: ('elmo_context+elmo_word', False, 'cosine', 'average', 150),
    8: ('elmo_context+elmo_word', False, 'manhattan', 'average', 150),
    # postevaluation
    9: ('elmo_context+elmo_word', True, 'manhattan', 'average', 235),

}

submissions_task22 = {
    1: ('w2v_context+inbound_deps', False, 'euclidean', 'ward', 2),
    2: ('inbound_deps', False, 'euclidean', 'ward', 2),
    3: ('w2v_word+inbound&outbound_deps', False, 'euclidean', 'ward', 6),
    4: ('elmo_context+elmo_word', 'LR'),
    5: ('elmo_context+elmo_word+inbound_deps+bool_pos', 'LR'),
    6: ('elmo_word+inbound_deps+bool_pos', 'LR'),
    7: ('w2v_context+bool_pos+123_pos', False, 'euclidean', 'single', 34),
    8: ('elmo_context+elmo_word+inbound_deps+bool_pos+123_pos', 'LR'),
    9: ('elmo_word+inbound_deps+bool_pos+123_pos', 'LR'),
    10: ('elmo_context+elmo_word+elmo_verb+inbound_deps+bool_pos+123_pos', 'LR'),
    11: ('elmo_context+elmo_word+elmo_verb+inbound_deps+bool_pos+123_pos', False, 'cosine', 'average', 34, 'joint',
         'concat'),
    12: ('elmo_context+elmo_word+elmo_verb+inbound_deps+bool_pos+123_pos', False, 'cosine', 'average', 34, 'joint',
         0.85),
    # postevaluation
    13: ('elmo_context+inbound_deps', 'euclidean, ward, 2'),
    14: ('w2v_context+w2v_word+inbound_deps+bool_pos', 'LR'),
    15: ('w2v_word+inbound_deps+bool_pos', 'LR'),
    16: ('w2v_context+w2v_word+inbound_deps+bool_pos+123_pos', 'LR'),
    17: ('w2v_word+inbound_deps+bool_pos+123_pos', 'LR'),
    18: ('w2v_context+w2v_word+w2v_verb+inbound_deps+bool_pos+123_pos', 'LR')
}

submissions = {
    1: (1, 1),
    2: (2, 2),
    3: (3, 3),
    4: (4, 4),
    5: (4, 5),
    6: (4, 6),
    7: (5, 7),
    8: (4, 8),
    9: (4, 9),
    10: (6, 10),
    11: (7,11),
    12: (8,12),
    13: (4, 10),
    14: (9, 18)
}
# ----------------------------------------------------------------------------
from make_textSubmission import task1Submission, task22Submission, task21Submission
from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('submission_id', type = int, choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                        help = "Which system submission? ")
# parser.add_argument('output_dir', help = "Where to put txt and zip file?")

args = parser.parse_args()

submission_id = args.submission_id
# submission_id=11

# output_dir = args.output_dir
output_dir = 'submission{}'.format(submission_id)


if not Path(output_dir).exists():
    os.system('mkdir '+output_dir)


task1_parameters, task22_parameters = submissions_task1[submissions[submission_id][0]], submissions_task22[
                                                                                            submissions[submission_id][1]]
task1Submission(task1_parameters, write=True, output_dir=output_dir)
task22Submission(task22_parameters, write=True, output_dir=output_dir)
task21Submission(output_dir)

os.system('chmod +x ./ziptask.sh')
os.chdir(output_dir)
print('Zip file is written to '+output_dir)
os.system('../ziptask.sh task-1.txt task-2.1.txt task-2.2.txt')

print('--------------------------------------------Evaluation---------------------------------------------')
print('\n----------Task1:\n')
os.system('java -cp ../scorer/EvaluationCodesSemEval2019Task2.jar semeval.run.Task1 ../semeval_data/gold/task-1.txt task-1.txt')

print('\n----------Task21:\n')
os.system('java -cp ../scorer/EvaluationCodesSemEval2019Task2.jar semeval.run.Task21 ../semeval_data/gold/task-2.1.txt task-2.1.txt')

print('\n----------Task22:\n')
os.system('java -cp ../scorer/EvaluationCodesSemEval2019Task2.jar semeval.run.Task22 ../semeval_data/gold/task-2.2.txt task-2.2.txt')
