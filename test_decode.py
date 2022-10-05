#!/usr/bin/env python3
from coq2vec import CoqTermRNNVectorizer, PAD_token, autoencoderBatchIter
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import itertools

vectorizer = CoqTermRNNVectorizer()
vectorizer.load_weights("weights/term2vec-weights-52.dat")
def display(s: str) -> str:
  input_seq = vectorizer.term_to_seq(s)
  encoded_term = vectorizer.seq_to_vector(input_seq)
  output_seq = vectorizer.vector_to_seq(encoded_term)
  output_term = vectorizer.output_seq_to_term(output_seq)
  if encoded_term.flatten().size(0) < 32:
      print(f"{s} -> {input_seq} -> {encoded_term.tolist()} -> {output_seq} -> {output_term}")
  else:
      # print(f"{s} ====>\n{input_seq} ====>\n{output_seq} ====>\n{output_term}")
      print(f"{s} =====>>>\n{output_term}")

device = "cuda" if torch.cuda.is_available else "cpu"
num_samples_testing = 10
with open("800000-samples-terms.txt", 'r') as f:
    samples = [l.strip().rstrip(".") for l in sorted(list(itertools.islice(f, num_samples_testing)), key=lambda l: min(vectorizer.term_seq_length(l), vectorizer.max_term_length), reverse=True)]
term_seqs = []
term_lengths = []
num_predicted = 0
num_correct = 0
for lidx, line in enumerate(samples):
    if lidx == 6:
        print(line)
    term_lengths.append(min(vectorizer.term_seq_length(line)+1, vectorizer.max_term_length))
    input_seq = vectorizer.term_to_seq(line)
    input_symbols = vectorizer.seq_to_symbol_list(input_seq)
    term_seqs.append(input_seq)
    output_seq = vectorizer.vector_to_seq(vectorizer.seq_to_vector(input_seq))
    output_symbols = vectorizer.seq_to_symbol_list(output_seq)[::-1]
    sample_correct = len([(t1, t2) for t1, t2 in
                          zip(input_symbols, output_symbols) if t1 == t2])
    # Account for matching the EOS token
    sample_predicted = len(input_symbols) + 1
    if len(input_symbols) == len(output_symbols):
        sample_correct += 1
    num_predicted += sample_predicted
    num_correct += sample_correct
    if lidx < 5:
        display(line.strip())
        print(f"Accuracy of sample: {sample_correct * 100 / sample_predicted:.2f}% ({sample_correct} / {sample_predicted})")
criterion = nn.NLLLoss(ignore_index=PAD_token)
term_batch = torch.tensor(term_seqs)
lengths_sorted, sorted_idxs = torch.tensor(term_lengths).sort(descending=True)
packed_term_batch = pack_padded_sequence(term_batch[sorted_idxs], lengths_sorted, batch_first=True)
test_loss, test_accuracy = autoencoderBatchIter(vectorizer.model, vectorizer._decoder,
                                                packed_term_batch.to(device),
                                                term_batch[sorted_idxs].to(device),
                                                lengths_sorted.to(device),
                                                criterion, 0.0, verbosity=1, model=vectorizer)
print(f"Accuracy, the training way: {test_accuracy * 100:.2f}%")
print(f"Overall accuracy from {num_samples_testing} samples: {num_correct * 100 / num_predicted:.2f}% ({num_correct} / {num_predicted})")
