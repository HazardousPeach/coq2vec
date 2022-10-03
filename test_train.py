#!/usr/bin/env python
from coq2vec import CoqTermRNNVectorizer
import random
import os
from typing import List
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

max_length=30
with open('800000-samples-terms.txt', 'r') as f:
    terms = list(tqdm(itertools.islice((l.strip() for l in tqdm(f, total=80000) if len(get_symbols(l)) < max_length - 1), 80000)))
vectorizer = CoqTermRNNVectorizer()
# 804857
os.makedirs("weights", exist_ok=True)
for epoch, _ in enumerate(vectorizer.train(terms, hidden_size=3712, learning_rate=0.32,
                                           n_epochs=60, batch_size=64, print_every=16,
                                           gamma=0.9, momentum=0.86,
                                           force_max_length=30, epoch_step=1,
                                           num_layers=2, teacher_forcing_ratio=0.38, verbosity=1)):
    weightspath = f"weights/term2vec-weights-{epoch}.dat"
    print(f"Saving weights to {weightspath}")
    vectorizer.save_weights(weightspath)
