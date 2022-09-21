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

from tqdm import tqdm

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
        self.symbol_mapping, self.token_vocab, self.model, self._decoder, self.max_term_length = torch.load(model_path)
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

        term_tensors = torch.LongTensor([
            normalize_sentence_length([self.symbol_mapping[symb]
                                       for symb in get_symbols(term)
                                       if symb in self.symbol_mapping],
                                      self.max_term_length,
                                      EOS_token) + [EOS_token]
            for term in tqdm(terms, desc="Tokenizing and normalizing")])

        data_batches = data.DataLoader(data.TensorDataset(term_tensors),
                                       batch_size=batch_size, num_workers=0,
                                       shuffle=True, pin_memory=True, drop_last=True)
        num_batches = int(term_tensors.size()[0] / batch_size)
        dataset_size = num_batches * batch_size

        encoder = maybe_cuda(EncoderRNN(len(self.token_vocab)+2, hidden_size, num_layers).to(self.device))
        decoder = maybe_cuda(DecoderRNN(hidden_size, len(self.token_vocab)+2, num_layers).to(self.device))

        optimizer = optim.SGD(itertools.chain(encoder.parameters(), decoder.parameters()),
                              lr=learning_rate)
        adjuster = scheduler.StepLR(optimizer, epoch_step,
                                    gamma=gamma)

        criterion = nn.NLLLoss()
        training_start=time.time()
        print("Training")
        for epoch in range(n_epochs):
            print("Epoch {} (learning rate {:.6f})".format(epoch, optimizer.param_groups[0]['lr']))
            epoch_loss = 0.
            for batch_num, (data_batch,) in enumerate(data_batches, start=1):
                optimizer.zero_grad()
                loss = autoencoderBatchLoss(encoder, decoder, maybe_cuda(data_batch), criterion)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if batch_num % print_every == 0:
                    items_processed = batch_num * batch_size + \
                      epoch * dataset_size
                    progress = items_processed / \
                      (dataset_size * n_epochs)
                    print("{} ({:7} {:5.2f}%) {:.4f}"
                          .format(timeSince(training_start, progress),
                                  items_processed, progress * 100,
                                  epoch_loss / batch_num))
            adjuster.step()
            self.model = encoder
            self._decoder = decoder
            pass
        pass
    def term_to_vector(self, term_text: str) -> torch.FloatTensor:
        assert self.symbol_mapping, "No loaded weights!"
        assert self.model, "No loaded weights!"
        term_sentence = get_symbols(term_text)
        term_tensor = maybe_cuda(torch.LongTensor(
            [self.symbol_mapping[symb] for symb in term_sentence
             if symb in self.symbol_mapping] + [EOS_token])).view(-1, 1)
        input_length = term_tensor.size(0)
        with torch.no_grad():
            device = "cuda" if use_cuda else "cpu"
            hidden = self.model.initHidden(1, device)
            cell = self.model.initCell(1, device)
            for ei in range(input_length):
                _, hidden, cell = self.model(term_tensor[ei], hidden, cell)
        return hidden.cpu().detach().numpy().flatten()
    def vector_to_term(self, term_vec: torch.FloatTensor) -> str:
        assert self.symbol_mapping, "No loaded weights!"
        assert self.model, "No loaded weights!"
        assert self._decoder
        assert self.max_term_length
        assert self.token_vocab
        assert term_vec.size() == torch.Size([self.model.hidden_size]), "Wrong dimensions for input"
        device = "cuda" if use_cuda else "cpu"
        self._decoder.to(device)
        output = ""
        with torch.no_grad():
            decoder_hidden = term_vec.to(device).view(1, 1, -1)
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_cell = self._decoder.initCell(1, device)
            for di in range(self.max_term_length):
                decoder_output, decoder_hidden, decoder_cell = self._decoder(decoder_input, decoder_hidden, decoder_cell)
                topv, topi = decoder_output.topk(1)
                next_char = topi.squeeze().detach()
                if next_char == EOS_token:
                    break
                decoder_input = next_char.view(1, 1)
                output += " " + self.token_vocab[decoder_input - 2]
        return output


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
        output, (hidden, cell) = self.lstm(embedded, (hidden,cell))
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
        self.softmax = nn.LogSoftmax(dim=1)
        self.num_layers = num_layers

    def forward(self, input, hidden, cell):
        batch_size = input.size(0)
        embedded = self.embedding(input).view(1, batch_size, -1)
        output, (hidden, cell) = self.lstm(F.relu(embedded), (hidden, cell))
        token_dist = self.softmax(self.out(output[0]))
        return token_dist, hidden, cell

    def initHidden(self,batch_size: int, device: str):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def initCell(self,batch_size: int, device: str):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

def autoencoderBatchLoss(encoder: EncoderRNN, decoder: DecoderRNN, data: torch.LongTensor, criterion: loss._Loss) -> torch.FloatTensor:
    batch_size = data.size(0)
    input_length = data.size(1)
    target_length = input_length
    device = "cuda" if use_cuda else "cpu"
    encoder_hidden = encoder.initHidden(batch_size, device)
    encoder_cell = encoder.initCell(batch_size, device)
    decoder_cell = decoder.initCell(batch_size, device)

    loss: torch.FloatTensor = maybe_cuda(torch.FloatTensor([0.]))
    for ei in range(input_length):
        encoder_output,encoder_hidden, encoder_cell = encoder(data[:,ei], encoder_hidden, encoder_cell)

    decoder_input = torch.tensor([[SOS_token]]*batch_size, device=device)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_cell = decoder(decoder_input, decoder_hidden, decoder_cell)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss = cast(torch.FloatTensor, loss + cast(torch.FloatTensor, criterion(decoder_output, data[:,di])))

    return loss


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
