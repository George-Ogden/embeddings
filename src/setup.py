from transformers import AutoTokenizer, AutoModelForMaskedLM
from .embedders.bert import BertForEmbedding

import torch

from typing import List

MODEL = "bert"

models = {
    "bert": {
        "tokenizer": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "embedder": BertForEmbedding.from_pretrained("bert-base-uncased"),
        "model": AutoModelForMaskedLM.from_pretrained("bert-base-uncased"),
    }
}

def get_tokenizer(model: str=MODEL):
    return models[model]["tokenizer"]

def get_model(model: str=MODEL):
    return models[model]["model"]

def get_embedder(model: str=MODEL):
    return models[model]["embedder"]

def embed(words: List[str], model: str=MODEL) -> torch.Tensor:
    inputs = get_tokenizer(model)(words, return_tensors="pt", padding=True)
    with torch.no_grad():
        return get_embedder(model)(**inputs)[:,1]