from transformers import pipeline
import torch

from typing import Any, Callable, ClassVar, Dict, List, Tuple, Union
from dataclasses import dataclass

from src.setup import embed, get_model, get_tokenizer

@dataclass
class Modification:
    input: str = "Rome is the capital of [MASK]."
    equation: str = "Rome := - Italy + France"
    symbols: ClassVar[Dict[str, Callable[[Any, Any], Any]]] = {
        "-": lambda x, y: x - y,
        "+": lambda x, y: x + y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y,
    }
    def parse(self) -> Tuple[str, List[Tuple[Union[str, float], Callable[[Any, Any], Any]]]]:
        initial, formula = self.equation.split(":=")
        tokens = formula.split()
        formula = [(float(token) if token.replace(".","",1).isdigit() else token, self.symbols[symbol]) for symbol, token in zip(tokens[0::2], tokens[1::2])]
        return initial.strip(), formula


def process(modification: Modification) -> str:
    model =  get_model()
    tokenizer = get_tokenizer()

    initial, formula = modification.parse()
    words, functions = zip(*formula)
    words = [initial] + list(words)
    floats = []
    for i, word in enumerate(words):
        if isinstance(word, float):
            floats.append((i, word))
            continue
        assert len(tokenizer(word).input_ids) == 3, "this only works with single-token embeddings"
    
    for i, _ in reversed(floats):
        words.pop(i)
    
    with torch.no_grad():
        embeddings = list(embed(words))
        for i, number in floats:
            embeddings.insert(i, number)
        embedding, operands = embeddings[0], embeddings[1:]
        for operand, function in zip(operands, functions):
            embedding = function(embedding, operand)

        index = tokenizer(initial).input_ids[1]
        model.bert.embeddings.word_embeddings._parameters["weight"][index] = embedding

    unmasker = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    return unmasker(modification.input)

if __name__ == "__main__":
    print(process(Modification()))