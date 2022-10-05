Coq2Vec
=======

A library for turning the text representation of coq terms into
fixed-width vectors for use in ML models. Does this by training an
autoencoder on a corpus of coq terms.

This repo is constructed from code by Abhishek Varghese, with
modifications by Alex Sanchez-Stern.



Example Usage
-------------

### Hyperparameter Search
```
>>> with open("terms.txt") as f:
....  terms = [l for l in f]
>>> tune_termrnn_hyperparameters(terms, n_epochs=20, batch_size=32,
                                 print_every=16, force_max_length=max_length,
								 epoch_step=1)
```

### Training
```
>>> max_length = 30
>>> vectorizer = CoqTermRNNVectorizer()
>>> with open("terms.txt") as f:
....  terms = [l for l in f]
>>> for epoch, _accuracy in enumerate(vectorizer.train(
                                        terms,
										hidden_size=3712, learning_rate=0.32,
                                        n_epochs=60, batch_size=64, print_every=16,
                                        gamma=0.9, momentum=0.86,
                                        force_max_length=30, epoch_step=1,
                                        num_layers=2, teacher_forcing_ratio=0.38,
										verbosity=1)):
....  vectorizer.save_weights(f"term2vec-weights-{epoch}.dat")
```

### Encoding
```
>>> import coq2vec
>>> vectorizer = coq2vec.CoqTermRNNVectorizer()
>>> vectorizer.load_weights("coq2vec/term2vec-weights-19.dat")
>>> vectorizer.term_to_vector("forall x: nat, x = x")
array([ 2.5957816e-06,  9.6319777e-01, -9.9500245e-01, ...,
       -7.0873748e-06, -5.7334816e-03, -2.8567877e-07], dtype=float32
```
