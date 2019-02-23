#Scorer for the SemEval2019 Task 2 (Unsuperivsed Lexical Frame Induction)

This code contains scorer for all the subtask. 
The code prints the following measures by comparing a reference (gold) annotaiton file with the secondary file that contains a clustering result. 
By default, the implemented code prints the following metric:

- BCubed Precision and Recall, and their harmonic mean (i.e., F1-Score);
- Purity, Inverse Purity, and their harmonic mean.


However, the jar file can be used also to obtain additional measure (for this you need to look into the code):
- Edit Distance (Pantel and Lin)
- Pair counting metrics: unsupervised precision and recall and their f1-score and Rand index and adjusted Rand index.
- Entropy based metrics including v-measure from the SemEval 2010 task on sense induction as well as Normalized Variation of information (Nguyen et al.)





#How to use the code?

Before using the code, make sure that you have java runtime 1.8 installed on your computer.
To use the scorer for subtask 1, open console and type:

$ java -cp EvaluationCodesSemEval2019Task2.jar semeval.run.Task1 path-to-gold-file path-to-submission-file -verbose

the last argument -verbose is optional, and tells the scorer to print baselines, too. Note that the order of arguments is important.


To use the scorer for subtask 2.1, use:
$ java -cp EvaluationCodesSemEval2019Task2.jar semeval.run.Task21 path-to-gold-file path-to-submission-file path-to-verb-gr-baseline -verbose
the last two arguments, i.e., path-to-submission-file and -verbose are optional. If -verbose is passed as an argument, then the scorer prints baselines too. 
In the case that the path-to-verb-gr-baseline-file is given, it will be included in the baseline results. Note that path-to-verb-gr-baseline can be replaced by any other file that contains valid evaluation records.

To use the scorer for subtask 2.2, use:
$ java -cp EvaluationCodesSemEval2019Task2.jar semeval.run.Task22 path-to-gold-file path-to-submission-file path-to-gr-baseline -verbose
Similar to subtask 2.1, the last two arguments are optional.

Some of the implementations are borrowed from the ELKI project (see the header of source files). 