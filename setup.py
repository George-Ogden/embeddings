from transformers import AutoTokenizer
from embedder import BertForEmbedding

import torch

from typing import List

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = BertForEmbedding.from_pretrained("bert-base-uncased")

def embed(words: List[str]) -> torch.Tensor:
    inputs = tokenizer(words, return_tensors="pt", padding=True)
    # assert inputs.input_ids.size(1) == 3
    with torch.no_grad():
        return model(**inputs)[:,1]