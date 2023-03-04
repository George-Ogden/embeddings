from transformers import AutoTokenizer
from embedders.bert import BertForEmbedding
from embedders.roberta import RobertaForEmbedding

import torch

from typing import List

MODEL = "bert"

models = {
    "bert": {
        "tokenizer": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "model": BertForEmbedding.from_pretrained("bert-base-uncased"),
    },
    "roberta": {
        "tokenizer": AutoTokenizer.from_pretrained("roberta-base"),
        "model": RobertaForEmbedding.from_pretrained("roberta-base"),
    }
}

def get_tokenizer(model: str=MODEL):
    return models[model]["tokenizer"]

def get_model(model: str=MODEL):
    return models[model]["model"]

def embed(words: List[str], model: str=MODEL) -> torch.Tensor:
    inputs = get_tokenizer(model)(words, return_tensors="pt", padding=True)
    # assert inputs.input_ids.size(1) == 3
    with torch.no_grad():
        return get_model(model)(**inputs)[:,1]