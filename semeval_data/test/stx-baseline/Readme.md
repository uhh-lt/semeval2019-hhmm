# List of syntactic dependencies in eud-baselines


The following relations based on Enhanced Universal Dependencies appeared in the baseline files for the argument clustering tasks (eud-b-task2.1.txt and eud-b-task2.2.txt):
 
```
nmod:between, nmod:with, nmod:by, nmod:of, nmod:next_to, nmod:off, nmod:through, xcomp, nmod:among, advmod, nmod:under, ccomp, nmod:tmod, nmod:to, nmod:below, nsubj, nmod:agent, nmod:according_to, nmod:over, iobj, nmod:as, nmod:outside, nmod:at, nmod:into, nmod:about, nmod:across, nmod, nmod:via, nmod:out_of, nmod:for, nmod:from, nmod:in, nmod:after, dobj, nmod:on
```

In addition to 

```
lscmpx

```
which indicates that the argument appeared at the left side of the target verb for which we could not find a direct syntactic relation, as well as


```
rscmpx

```

which states a similar info for arguments at the right side of the target verb.

These relations are pulled out from automatically parsed sentences using the Stanford parser. However, note that the parser was pre-trained on PTB WSJ (presumably the parser shows better performance than the time it is applied to out of domain sentences). 