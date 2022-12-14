#!/usr/bin/env python
from coq2vec import CoqTermRNNVectorizer, tune_termrnn_hyperparameters
from typing import List
import random
import re
from tqdm import tqdm
import itertools

symbols_regexp = (r',|(?::>)|(?::(?!=))|(?::=)|\)|\(|;|@\{|~|\+{1,2}|\*{1,2}|&&|\|\||'
                  r'(?<!\\)/(?!\\)|/\\|\\/|(?<![<*+-/|&])=(?!>)|%|(?<!<)-(?!>)|'
                  r'<-|->|<=|>=|<>|\^|\[|\]|(?<!\|)\}|\{(?!\|)')
def get_symbols(string: str) -> List[str]:
    return [word for word in re.sub(
        r'(' + symbols_regexp + ')',
        r' \1 ', string).split()
            if word.strip() != '']

max_length = 20
num_terms = 20484

with open('800000-samples-terms.txt', 'r') as f:
    terms = list(tqdm(itertools.islice((l.strip() for l in f if len(get_symbols(l)) < max_length - 1),num_terms), total=num_terms))

tune_termrnn_hyperparameters(terms, n_epochs=20, batch_size=32,
                             print_every=16, force_max_length=max_length, epoch_step=1)
