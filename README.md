# semeval2019-hhmm

Installations
-----------

```
pip install -r requirements.txt
```
Other Requirements
-----------

```
donwload and extract google news model into: '/input/models/GoogleNews-vectors-negative300.bin'

Source: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit 

```
make_zip_submission
-----------
Main file to create and evaluate a system submission
```
python make_zip_submission submission_id output_dir

There were total 14 submissions (including 1 post-competition submission), so
submission_ids = [1, 2, 3,...14]

```

resource_helper
-----------
To speedup processing, write some useful files to 'input' directory
```
Contains method to write:
 a) csv files from txt inputs
 b) most commonly used vectors like 'context' and 'word' etc.
 c) dump word2vec and elmo models to speedup the processing
 
```