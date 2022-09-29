#!/usr/bin/env python
from coq2vec import CoqTermRNNVectorizer
import random
import os
with open('/home/asanchezster_umass_edu/work/coq2vec/terms.txt', 'r') as f:
    terms = [l.strip() for l in f]
vectorizer = CoqTermRNNVectorizer()
# 804857
os.makedirs("weights", exist_ok=True)
for epoch, _ in enumerate(vectorizer.train(random.sample(terms, 10484), hidden_size=1780, learning_rate=0.15, 
                                           n_epochs=30, batch_size=16, print_every=16, 
                                           gamma=0.6, momentum=0.4,
                                           force_max_length=30, epoch_step=1, 
                                           num_layers=1, verbosity=1)):
    weightspath = f"weights/term2vec-weights-{epoch}.dat"
    print(f"Saving weights to {weightspath}")
    vectorizer.save_weights(weightspath)
