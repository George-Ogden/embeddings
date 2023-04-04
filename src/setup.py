from transformers import AutoTokenizer, AutoModelForMaskedLM
from .embedders.bert import BertForEmbedding

import torch

from typing import List

# define default model
MODEL = "bert"

# define models, tokenizers, and embedders (currently just bert)
models = {
    "bert": {
        "tokenizer": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "embedder": BertForEmbedding.from_pretrained("bert-base-uncased"),
        "model": AutoModelForMaskedLM.from_pretrained("bert-base-uncased"),
    }
}


def get_tokenizer(model: str = MODEL) -> AutoTokenizer:
    return models[model]["tokenizer"]


def get_model(model: str = MODEL) -> AutoModelForMaskedLM:
    return models[model]["model"]


def get_embedder(model: str = MODEL) -> BertForEmbedding:
    return models[model]["embedder"]


def embed(words: List[str], model: str = MODEL) -> torch.Tensor:
    """embed a list of words

    Args:
        words (List[str]): list of words to tokenise and embed
        model (str, optional): name of the model to use. Defaults to MODEL.

    Returns:
        torch.Tensor: list of embeddings
    """
    # embed the words
    inputs = get_tokenizer(model)(words, return_tensors="pt", padding=True)
    with torch.no_grad():
        # return index 1 of the embeddings (excluding [CLS] and [SEP] tokens)
        return get_embedder(model)(**inputs)[:, 1]
