# SemEval 2019 Task 2 on Frame Induction 


## Summary:

This resource contains material used for the evaluation in Task 2 of SemEval2019: a set of typed feature structures, the so called lexical-semantic frames annotated manually based on the SemEval2019 Task2 annotation guidelines 1.0. For more information about the CodaLab pages for the task at https://competitions.codalab.org/competitions/19159
 
This resource is free as long as it is used for research and that its use is acknowledged (citation records to be provided) and under the conditions stated in the attached licenses (see files in the `licenses` directory). For adaptations, redistributions, and commercial purposes, please contact LDC (`ldc@ldc.upenn.edu`). [further information will be available gradually]  

## Directory structure/content:

### ./licenses

The directory contains licenses and copyright notices including links to projects and resources that have influenced the development of this resource (see `readme.license.txt`).

### ./dep-stx

This directory contains three files in a *CoNLL-X* format. These files contain tokenized, part-of-speech tagged, dependency-parsed sentences which may have been used during the annotation. **Most importantly**, the assigned IDs to the sentences and the position of tokens in these files are used as the main reference for adding our annotation layer.
 
The format of sentence IDs is similar to the PSD resource (SemEval 2014 Task 8). Each sentence id starts with a `#` followed by a unique eight-digit number in the form of `2SSDDIII`, in which '2' is a constant, followed by a two-digit section code (`SS`), a two-digit document identifier for the section (`DD`), and a three-digit (`III`) that identifies the entry/sentence within each document. For example, identifier `#20001001` is the first sentence in the first file of section 01: Pierre Vinken, 61 years old ... . This identifier can be used to fetch gold annotations such as parses for the sentences from/for PTB (for the task participants, the treebank can be acquired free of charge from LDC). In all the files, the empty nodes in PTB (marked by *- followed by a number) are removed. All files contain most sentences up to section 22 (23 and 24 are only in the automatically generated files).

The three files are:  
* `./dep-stx/assumed-gold-.conll.txt` which contains gold part-of-speech tags and dependency parses according to the "Stanford base dependencies formalism" that are derived from the gold constituent parses in the Treebank using the Stanford conversion tool (for more information see the SDP resource). Note that the conversion problem is not 100% perfect and small errors in the asserted dependencies between words can be expected.
* `./dep-stx/pos-gold-dep-auto.conll.txt` which contains sentences parsed *automatically* using the pre-trained Stanford dependency parser in the "enhanced universal dependency formalism". For this, the parser is fed with the Gold part-of-speech tags from PTB. And,
* `./dep-stx/pos-auto-dep-auto.conll.txt` which contains sentences part-of-speech tagged and parsed automatically (though the pos tagger was pre-trained on PTB).

You are welcome to use any syntactic parser of any formalism (e.g., HPSG) as input to your system.

### ./dev
This directory contains the development/trial data are made available for the participants of SemEval2019 Task 2 prior to the evaluation phase:
* a small sample of gold annotations for each of the subtasks, namely `./dev/task-1.txt` (for subtask 1), `./dev/task-2.1.txt` (for subtask 2.1), and `./dev/task-2.2.txt` (for subtask 2.2). 
* a larger sample of annotations derived semi-automatically by merging annotations previously carried out in different projects. These automatic annotations are placed in `./dev/auto`, one file per subtask. The instances in these files are sampled randomly from the final test set for the task. NB: although the automatic annotation are derived from manual annotations (e.g. SemLink, VerbNet, PSD, etc.) they may not be correct or complete in any way. 

### ./test 	
This directory contains the test data used for the blind evaluation of participating systems. The structure is similar to the `dev` folder: one file per subtask. All the gold labels are replaced by `unkn` token, which will be replaced by the gold annotations after the evaluation period. 

The fold also contains a directory named `stx-baseline`, containing simple baseline files based on the output of automatic parses (see the `test/stx-baseline/Readme.md`).

Note that the number of arguments to be clustered for Task 2.1 and Task 2.2 are different (some of the core elements of FN for Task 2.1 are considered adjunct for Semantic Role labeling based on VerbNet)

**NB: this is a work in progress and additional resources are provided according to the SemEval 2019 time-line.**
	
## Contact: 
for questions or additional information, send email to `semeval-2019-task-2-organizers@googlegroups.com` and/or `semeval_frameinduction@googlegroups.com`.


## Acknowledgment: 
This resource has been developed through the support of DFG Collaborative Research Center 991 at Heinrich-Heine Düsseldorf University, Chair Computational Linguistics, Prof. Dr. Laura Kallmeyer. We would like to thank FrameNet team for their continues support through the annotation task, as well as LDC for their generous help by providing access to PTB 3.0 and related resources.
