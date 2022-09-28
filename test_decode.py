from coq2vec import CoqTermRNNVectorizer
import torch
vectorizer = CoqTermRNNVectorizer()
vectorizer.load_weights("term2vec-weights-29.dat")
def display(s: str) -> str:
  input_seq = vectorizer.term_to_seq(s)
  encoded_term = vectorizer.seq_to_vector(input_seq)
  output_seq = vectorizer.vector_to_seq(encoded_term)
  output_term = vectorizer.seq_to_term(output_seq)
  if encoded_term.flatten().size(0) < 32:
      print(f"{s} -> {input_seq} -> {encoded_term.tolist()} -> {output_seq} -> {output_term}")
  else:
      print(f"{s} -> {input_seq} -> {output_seq} -> {output_term}")
display("n = n")
display("nat")
