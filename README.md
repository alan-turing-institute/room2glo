# room2glo


Create synthetic dataset
------------------------

**1.** run get_freqs_and_wordnet_stats.py to create a file containing frequency counts and wordnet senses for all words in the vocabulary for a chosen month.

```python
usage:  get_freqs_and_wordnet_stats.py -i INPUT_FILEPATH -o OUTPUT_DIR
```
(input file should be gzipped tab-separated values)


**2.** run design_pseudowords.py to create two dictionaries (which will be stored in json format) specifying pseudowords, the real words which represent their different possible senses, and the probabilities associated with each sense at each timestep.

```python
usage: design_pseudowords.py -i INPUT_FILEPATH -o OUTPUT_ROOTDIR 
                             -sy START_YEAR -sm START_MONTH 
                             -ey END_YEAR -em END_MONTH 
                             -mf MIN_FREQ -ms MAX_N_SENSES
```


  
**3.** run create_synthetic_dataset.py to create the synthetic dataset. (add the option `-sp 1` to subsample from the original month rather than duplicating it)

```python
usage: create_synthetic_dataset.py -i INPUT_FILEPATH -o OUTPUT_ROOTDIR
                                   -c CONTEXT_WORD_DICT_FILEPATH 
                                   -p PSEUDOWORD_DICT_FILEPATH
                                   [-s SUBSAMPLING] [-sp SUBSAMPLING_PERCENT]
                                   -sy START_YEAR -sm START_MONTH 
                                   -ey END_YEAR -em END_MONTH 
```

