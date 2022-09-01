from typing import List, TypeVar, Dict, Optional, Union, overload, cast
import re
import sys
import contextlib
import pickle
from pathlib import Path

import torch
from torch import nn

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

class CoqRNNVectorizer:
    symbol_mapping: Optional[Dict[str, int]]
    model: Optional['EncoderRNN']
    def __init__(self) -> None:
        self.symbol_mapping = None
        self.model = None
        pass
    def load_weights(self, language_symbols_path: Path,
                     model_path: Path) -> None:
        self.symbol_mapping = pickle.load(open(str(language_symbols_path), 'rb'))
        device = "cuda" if use_cuda else "cpu"
        self.model = maybe_cuda(torch.load(str(model_path), map_location=device))
        pass
    def save_weights(self):
        raise NotImplementedError()
        pass
    def train(self):
        raise NotImplementedError()
        pass
    def term_to_vector(self, term_text: str):
        assert self.symbol_mapping
        assert self.model
        term_sentence = get_symbols(term_text)
        term_tensor = maybe_cuda(torch.LongTensor(
            [self.symbol_mapping[symb] for symb in term_sentence
             if symb in self.symbol_mapping] + [EOS_token])).view(-1, 1)
        input_length = term_tensor.size(0)
        with torch.no_grad():
            device = "cuda" if use_cuda else "cpu"
            hidden = self.model.initHidden(device)
            cell = self.model.initCell(device)
            for ei in range(input_length):
                _, hidden, cell = self.model(term_tensor[ei], hidden, cell)
        return hidden.cpu().detach().numpy().flatten()

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden, cell):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, (hidden, cell) = self.lstm(output, (hidden,cell))
        return output, hidden, cell

    def initHidden(self,device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def initCell(self,device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

symbols_regexp = (r',|(?::>)|(?::(?!=))|(?::=)|\)|\(|;|@\{|~|\+{1,2}|\*{1,2}|&&|\|\||'
                  r'(?<!\\)/(?!\\)|/\\|\\/|(?<![<*+-/|&])=(?!>)|%|(?<!<)-(?!>)|'
                  r'<-|->|<=|>=|<>|\^|\[|\]|(?<!\|)\}|\{(?!\|)')
def get_symbols(string: str) -> List[str]:
    return [word for word in re.sub(
        r'(' + symbols_regexp + ')',
        r' \1 ', string).split()
            if word.strip() != '']

@overload
def maybe_cuda(component: nn.Module) -> nn.Module:
    ...

@overload
def maybe_cuda(component: torch.Tensor) -> torch.Tensor:
    ...

def maybe_cuda(component):
    if use_cuda:
        return component.to(device=torch.device(cuda_device))
    else:
        return component
