import pytest
import torch
import torch.nn as nn
from typing import Any

from gemma4_lab.inference.hf_local import GemmaLocal
from gemma4_lab.config import Settings
from gemma4_lab.interp.recorder import ActivationRecorder

class TinyLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.eye(d_model))
        
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> tuple:
        # Multiply by weight and add some non-linearity
        return (torch.matmul(hidden_states, self.weight),)
        
    def register_forward_hook(self, hook: Any) -> Any:
        return super().register_forward_hook(hook)

class TinyLanguageModel(nn.Module):
    def __init__(self, n_layers: int, d_model: int):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int = 100, d_model: int = 16, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Simulating AutoModelForCausalLM structure (model.layers)
        self.model = TinyLanguageModel(n_layers, d_model)
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids: torch.Tensor, **kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)[0]
        x = self.model.norm(x)
        logits = self.lm_head(x)
        
        from collections import namedtuple
        Output = namedtuple("Output", ["logits"])
        return Output(logits=logits)
        
    def get_input_embeddings(self):
        return self.embed_tokens
        
    def get_output_embeddings(self):
        return self.lm_head

class DummyBatchEncoding(dict):
    def to(self, device):
        return {k: v.to(device) for k, v in self.items()}

class DummyTokenizer:
    def __init__(self):
        self.vocab_size = 100
        
    def __call__(self, text: str, return_tensors: str = "pt", **kwargs):
        # Fake tokenization based on length for deterministic tests
        length = max(2, len(text) // 5)
        # return arange so the last token differs by sequence length
        ids = torch.arange(length, dtype=torch.long).unsqueeze(0)
        return DummyBatchEncoding({"input_ids": ids})

    def decode(self, ids: list[int] | torch.Tensor, **kwargs) -> str:
        return "dummy text"
        
    def apply_chat_template(self, messages: list[dict], **kwargs) -> str:
        return messages[0]["content"]

@pytest.fixture
def tiny_model():
    model = TinyTransformer(vocab_size=100, d_model=16, n_layers=2)
    model.eval()
    return model

@pytest.fixture
def dummy_tokenizer():
    return DummyTokenizer()

@pytest.fixture
def mock_recorder(tiny_model, dummy_tokenizer):
    """
    Creates an ActivationRecorder backed by our tiny model, ensuring
    sociable testing (no mocks of the PyTorch mechanics).
    """
    gemma = GemmaLocal.__new__(GemmaLocal)
    gemma._model = tiny_model
    gemma._tokenizer = dummy_tokenizer
    gemma._device = "cpu"
    gemma._load = lambda: None
    
    return ActivationRecorder(gemma)
