Coq2Vec
=======

A library for turning the text representation of coq terms into
fixed-width vectors for use in ML models. Does this by training an
autoencoder on a corpus of coq terms.

This repo is constructed from code by Abhishek Vargas.

Example Usage
-------------
```
>>> import coq2vec
>>> from coq2vec import EncoderRNN
>>> vectorizer = coq2vec.CoqRNNVectorizer()
>>> vectorizer.load_weights("coq2vec/symbols_dict.pickle", "coq2vec/encoder_symbols.model")
>>> vectorizer.term_to_vector("forall x: nat, x = x")
array([ 2.5957816e-06,  9.6319777e-01, -9.9500245e-01, ...,
       -7.0873748e-06, -5.7334816e-03, -2.8567877e-07], dtype=float32
```
