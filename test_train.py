#!/usr/bin/env python
from coq2vec import CoqTermRNNVectorizer
import random
import os
with open('/home/alex/coq2vec/terms.txt', 'r') as f:
    terms = [l.strip() for l in f]
vectorizer = CoqTermRNNVectorizer()
# 804857
os.makedirs("weights", exist_ok=True)
for epoch, _ in enumerate(vectorizer.train(random.sample(terms, 80000), hidden_size=512, learning_rate=0.81,
                                           n_epochs=12, batch_size=64, print_every=32,
                                           gamma=0.82, momentum=0.86,
                                           force_max_length=30, epoch_step=1,
                                           num_layers=1, teacher_forcing_ratio=0.3, verbosity=1)):
    weightspath = f"weights/term2vec-weights-{epoch}.dat"
    print(f"Saving weights to {weightspath}")
    vectorizer.save_weights(weightspath)
