from typing import (List, TypeVar, Dict, Optional, Union,
                    overload, cast, Set, NamedTuple)
import re
import sys
import contextlib
import pickle
import itertools
import time
from pathlib import Path

import torch
from torch import optim
from torch import nn
import torch.nn.modules.loss as loss
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim.lr_scheduler as scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import numpy as np

class Obligation(NamedTuple):
    hypotheses: List[str]
    goal: str

class ProofContext(NamedTuple):
    fg_goals: List[Obligation]
    bg_goals: List[Obligation]
    shelved_goals: List[Obligation]
    given_up_goals: List[Obligation]

class DummyFile:
    def write(self, x): pass
    def flush(self): pass

@contextlib.contextmanager
def silent():
    save_stderr = sys.stderr
    save_stdout = sys.stdout
    sys.stderr = DummyFile()
    sys.stdout = DummyFile()
    try:
        yield
    finally:
        sys.stderr = save_stderr
        sys.stdout = save_stdout

with silent():
    use_cuda = torch.cuda.is_available()
cuda_device = "cuda:0"
EOS_token = 1
SOS_token = 0

class CoqContextVectorizer:
    term_vectorizer: Optional['CoqTermRNNVectorizer']
    hypotheses_encoder: Optional['EncoderRNN']
    hypotheses_decoder: Optional['DecoderRNN']
    max_num_hypotheses: Optional[int]

    def __init__(self) -> None:
        self.term_encoder = None
        self.hypotheses_encoder = None
        self.hypotheses_decoder = None
        self.max_num_hypotheses = None
    def load_weights(self, model_path: Union[Path, str]) -> None:
        if isinstance(model_path, str):
            model_path = Path(model_path)
        with model_path.open('rb') as f:
            self.term_encoder, self.hypotheses_encoder,\
                self.hypotheses_decoder, self.max_num_hypotheses = torch.load(f)
    def save_weights(self, model_path: Union[Path, str]):
        if isinstance(model_path, str):
            model_path = Path(model_path)
        with model_path.open('wb') as f:
            torch.save((self.term_encoder, self.hypotheses_encoder,
                        self.hypotheses_decoder, self.max_num_hypotheses), f)
    def train(self, contexts: List[ProofContext],
              hidden_size: int, learning_rate: float, n_epochs: int,
              batch_size: int, print_every: int, gamma: float,
              force_max_length: Optional[int] = None, epoch_step: int = 1,
              num_layers: int = 1, allow_non_cuda: bool = False) -> None:
        pass

class CoqTermRNNVectorizer:
    symbol_mapping: Optional[Dict[str, int]]
    token_vocab: Optional[List[str]]
    model: Optional['EncoderRNN']
    _decoder: Optional['DecoderRNN']
    max_term_length: Optional[int]
    def __init__(self) -> None:
        self.symbol_mapping = None
        self.token_vocab = None
        self.model = None
        self._decoder = None
        self.max_term_length = None
        pass
    def load_weights(self, model_path: Union[Path, str]) -> None:
        if isinstance(model_path, str):
            model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.symbol_mapping, self.token_vocab, self.model, self._decoder, self.max_term_length = \
            torch.load(model_path, map_location=self.device)
    def save_weights(self, model_path: Union[Path, str]):
        if isinstance(model_path, str):
            model_path = Path(model_path)
        with model_path.open('wb') as f:
            torch.save((self.symbol_mapping, self.token_vocab, self.model, self._decoder, self.max_term_length), f)
        pass
    def train(self, terms: List[str],
              hidden_size: int, learning_rate: float, n_epochs: int,
              batch_size: int, print_every: int, gamma: float,
              force_max_length: Optional[int] = None, epoch_step: int = 1,
              num_layers: int = 1, allow_non_cuda: bool = False) -> None:
        assert use_cuda or allow_non_cuda, "Cannot train on non-cuda device unless passed allow_non_cuda"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        token_set: Set[str] = set()
        max_length_so_far = 0
        for term in tqdm(terms, desc="Getting symbols"):
            for symbol in get_symbols(term):
                token_set.add(symbol)
            max_length_so_far = max(len(get_symbols(term)), max_length_so_far)

        self.token_vocab = list(token_set)
        self.symbol_mapping = {}
        for idx, symbol in enumerate(self.token_vocab, start=2):
            self.symbol_mapping[symbol] = idx
        if force_max_length:
            self.max_term_length = min(force_max_length, max_length_so_far)
        else:
            self.max_term_length = max_length_so_far

        term_tensors = torch.LongTensor([self.term_to_seq(term)
            for term in tqdm(terms, desc="Tokenizing and normalizing")])

        dataset_size = len(terms)
        split_ratio = 0.05
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        split = int((dataset_size * split_ratio) / batch_size) * batch_size
        train_indices, val_indices = indices[split:], indices[:split]
        valid_batch_size = max(batch_size // 2, 1)
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        num_batches = int((dataset_size - split) / batch_size)
        num_batches_valid = int(split / valid_batch_size)

        train_dataset_size = num_batches * batch_size

        valid_data_batches = data.DataLoader(data.TensorDataset(term_tensors),
                                             batch_size=valid_batch_size, num_workers=0,
                                             sampler=valid_sampler, pin_memory=True, drop_last=True)
        data_batches = data.DataLoader(data.TensorDataset(term_tensors),
                                       batch_size=batch_size, num_workers=0,
                                       sampler=train_sampler, pin_memory=True, drop_last=True)

        encoder = maybe_cuda(EncoderRNN(len(self.token_vocab)+2, hidden_size, num_layers).to(self.device))
        self.model = encoder
        decoder = maybe_cuda(DecoderRNN(hidden_size, len(self.token_vocab)+2, num_layers).to(self.device))
        self._decoder = decoder
        optimizer = optim.SGD(itertools.chain(encoder.parameters(), decoder.parameters()),
                              lr=learning_rate)
        adjuster = scheduler.StepLR(optimizer, epoch_step,
                                    gamma=gamma)

        criterion = nn.NLLLoss()
        training_start=time.time()
        writer = SummaryWriter()
        print("Training")
        for epoch in range(n_epochs):
            print("Epoch {} (learning rate {:.6f})".format(epoch, optimizer.param_groups[0]['lr']))
            epoch_loss = 0.
            for batch_num, (data_batch,) in enumerate(data_batches, start=1):
                optimizer.zero_grad()
                loss, accuracy = autoencoderBatchIter(encoder, decoder, maybe_cuda(data_batch), criterion)
                writer.add_scalar("Batch loss/train", loss, epoch * num_batches + batch_num)
                writer.add_scalar("Batch accuracy/train", accuracy, epoch * num_batches + batch_num)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if batch_num % print_every == 0:
                    items_processed = batch_num * batch_size + \
                      epoch * train_dataset_size
                    progress = items_processed / \
                      (train_dataset_size * n_epochs)
                    print("{} ({:7} {:5.2f}%) {:.4f}"
                          .format(timeSince(training_start, progress),
                                  items_processed, progress * 100,
                                  epoch_loss / batch_num))
            with torch.no_grad():
                valid_accuracy = maybe_cuda(torch.FloatTensor([0.]))
                valid_loss = maybe_cuda(torch.FloatTensor([0.]))
                for (valid_data_batch,) in valid_data_batches:
                   batch_loss, batch_accuracy = autoencoderBatchIter(encoder, decoder, maybe_cuda(valid_data_batch), criterion)
                   valid_loss = cast(torch.FloatTensor, valid_loss + batch_loss)
                   valid_accuracy = cast(torch.FloatTensor, valid_accuracy + batch_accuracy)
            writer.add_scalar("Loss/valid", valid_loss / num_batches_valid,
                              epoch * num_batches + batch_num)
            writer.add_scalar("Accuracy/valid", valid_accuracy / num_batches_valid,
                              epoch * num_batches + batch_num)
            print(f"Validation loss: {valid_loss.item() / num_batches_valid:.4f}; "
                  f"Validation accuracy: {valid_accuracy.item() / num_batches_valid:.2f}%")

            adjuster.step()
            self.model = encoder
            self._decoder = decoder
            pass
        pass
    def term_to_seq(self, term_text: str) -> List[int]:
        return normalize_sentence_length([self.symbol_mapping[symb]
                                          for symb in get_symbols(term_text)
                                          if symb in self.symbol_mapping],
                                         self.max_term_length,
                                         EOS_token) + [EOS_token]
    def seq_to_term(self, seq: List[int]) -> str:
        output_symbols = []
        for item in seq:
            if item == EOS_token:
                break
            assert item >= 2
            output_symbols.append(self.token_vocab[item - 2])
        return " ".join(output_symbols)
    def term_to_vector(self, term_text: str) -> torch.FloatTensor:
        return self.seq_to_vector(self.term_to_seq(term_text))

    def seq_to_vector(self, term_seq: List[int]) -> torch.FloatTensor:
        assert self.symbol_mapping, "No loaded weights!"
        assert self.model, "No loaded weights!"
        term_tensor = maybe_cuda(torch.LongTensor([term_seq]))
        input_length = term_tensor.size(1)
        with torch.no_grad():
            device = "cuda" if use_cuda else "cpu"
            hidden = self.model.initHidden(1, device)
            cell = self.model.initCell(1, device)
            for ei in range(input_length):
                _, hidden, cell = self.model(term_tensor[:,ei], hidden, cell)
        return hidden.cpu()
    def vector_to_term(self, term_vec: torch.FloatTensor) -> str:
        return self.seq_to_term(self.vector_to_seq(term_vec))
    def vector_to_seq(self, term_vec: torch.FloatTensor) -> List[int]:
        assert self.symbol_mapping, "No loaded weights!"
        assert self.model, "No loaded weights!"
        assert self._decoder
        assert self.max_term_length
        assert self.token_vocab
        assert term_vec.size() == torch.Size([1, self.model.num_layers, self.model.hidden_size]), f"Wrong dimensions for input {term_vec.size()}"
        device = "cuda" if use_cuda else "cpu"
        self._decoder.to(device)
        output = ""
        with torch.no_grad():
            decoder_hidden = term_vec.to(device)
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_cell = self._decoder.initCell(1, device)
            for di in range(self.max_term_length):
                decoder_output, decoder_hidden, decoder_cell = self._decoder(decoder_input, decoder_hidden, decoder_cell)
                topv, topi = decoder_output.topk(1)
                next_char = topi.view(1).detach()
                output_seq.append(next_char.item())
                decoder_input = next_char
        return output_seq


class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.num_layers = num_layers

    def forward(self, input: torch.LongTensor, hidden: torch.FloatTensor,
                cell: torch.FloatTensor):
        batch_size = input.size(0)
        embedded = self.embedding(input).view(1, batch_size, self.hidden_size)
        output, (hidden, cell) = self.lstm(F.relu(embedded), (hidden,cell))
        return output, hidden, cell

    def initHidden(self,batch_size: int, device: str):
        return maybe_cuda(torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))

    def initCell(self,batch_size: int, device: str):
        return maybe_cuda(torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, num_layers: int = 1) -> None:
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.num_layers = num_layers

    def forward(self, input, hidden, cell):
        batch_size = input.size(0)
        embedded = self.embedding(input).view(1, batch_size, self.hidden_size)
        output, (hidden, cell) = self.lstm(F.relu(embedded), (hidden, cell))
        token_dist = self.softmax(self.out(output))
        return token_dist, hidden, cell

    def initHidden(self,batch_size: int, device: str):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def initCell(self,batch_size: int, device: str):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

def autoencoderBatchIter(encoder: EncoderRNN, decoder: DecoderRNN, data: torch.LongTensor,
                         criterion: loss._Loss, verbose=False) -> torch.FloatTensor:
    batch_size = data.size(0)
    input_length = data.size(1)
    target_length = input_length
    device = "cuda" if use_cuda else "cpu"
    encoder_hidden = encoder.initHidden(batch_size, device)
    encoder_cell = encoder.initCell(batch_size, device)
    decoder_cell = decoder.initCell(batch_size, device)

    loss: torch.FloatTensor = maybe_cuda(torch.FloatTensor([0.]))
    accuracy_sum: torch.FloatTensor = maybe_cuda(torch.FloatTensor([0.]))
    for ei in range(input_length):
        encoder_output,encoder_hidden, encoder_cell = encoder(data[:,ei], encoder_hidden, encoder_cell)

    decoder_input = torch.tensor([[SOS_token]]*batch_size, device=device)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_cell = decoder(decoder_input, decoder_hidden, decoder_cell)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.view(batch_size).detach()

        loss = cast(torch.FloatTensor, loss + cast(torch.FloatTensor, criterion(decoder_output, data[:,di])))
        if verbose:
            print(f"comparing {decoder_input} to {data[:,di]}")
        accuracy_sum += torch.sum(decoder_input == data[:,di])

    if verbose:
        print(f"Accuracy sum is {accuracy_sum}")
        print(f"Accuracy is {accuracy_sum / (batch_size * target_length)}")

    return loss / (batch_size * target_length), accuracy_sum / (batch_size * target_length)


symbols_regexp = (r',|(?::>)|(?::(?!=))|(?::=)|\)|\(|;|@\{|~|\+{1,2}|\*{1,2}|&&|\|\||'
                  r'(?<!\\)/(?!\\)|/\\|\\/|(?<![<*+-/|&])=(?!>)|%|(?<!<)-(?!>)|'
                  r'<-|->|<=|>=|<>|\^|\[|\]|(?<!\|)\}|\{(?!\|)')
def get_symbols(string: str) -> List[str]:
    return [word for word in re.sub(
        r'(' + symbols_regexp + ')',
        r' \1 ', string).split()
            if word.strip() != '']

T1 = TypeVar('T1', bound=nn.Module)
T2 = TypeVar('T2', bound=torch.Tensor)
@overload
def maybe_cuda(component: T1) -> T1:
    ...

@overload
def maybe_cuda(component: T2) -> T2:
    ...

def maybe_cuda(component):
    if use_cuda:
        return component.to(device=torch.device(cuda_device))
    else:
        return component

def normalize_sentence_length(sentence: List[int], target_length: int, fill_value: int) -> List[int]:
    if len(sentence) > target_length:
        return sentence[:target_length]
    elif len(sentence) < target_length:
        return sentence + [fill_value] * (target_length - len(sentence))
    else:
        return sentence

def timeSince(since : float, percent : float) -> str:
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return "{} (- {})".format(asMinutes(s), asMinutes(rs))

def asMinutes(s : float) -> str:
    m = int(s / 60)
    s -= m * 60
    return "{:3}m {:5.2f}s".format(m, s)
