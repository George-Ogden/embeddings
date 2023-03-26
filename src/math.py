from transformers import pipeline
import torch
import json

from typing import Any, Callable, ClassVar, Dict, List, Tuple, Optional, Union
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
        # convert every pair of tokens into a tuple of (word, function)
        formula = [
            (
                float(token) # keep floats as floats
                if token.replace(".","",1).isdigit()
                else token,
                self.symbols[symbol] # substitute symbol for its function
            )
            for symbol, token in zip(tokens[0::2], tokens[1::2])
        ]
        return initial.strip(), formula

def run(input: str, equation: Optional[str] = None):
    print(
        json.dumps(
            process(
                Modification(
                    input=input,
                    equation=equation
                )
            ),
            indent=True
        )
    )

def process(modification: Modification) -> str:
    # get model and tokenizer
    model =  get_model()
    tokenizer = get_tokenizer()

    if modification.equation:
        # parse equation
        initial, formula = modification.parse()

        # convert formula to code
        words, functions = zip(*formula)

        words = [initial] + list(words)
        floats = []
        for i, word in enumerate(words):
            if isinstance(word, float):
                # ignore floats
                floats.append((i, word))
                continue
            assert len(tokenizer(word).input_ids) == 3, "this only works with single-token embeddings"
        
        # remove the floats from the list of words
        for i, _ in reversed(floats):
            words.pop(i)
        
        with torch.no_grad():
            # convert words to embeddings
            embeddings = list(embed(words))
            # put the floats back in
            for i, number in floats:
                embeddings.insert(i, number)
            
            # apply the operations on the embeddings
            embedding = 0.
            for operand, function in zip(embeddings[1:], functions):
                embedding = function(embedding, operand)
            embedding += embeddings[0]

            # update the embedding in the embedding layer of the model
            index = tokenizer(initial).input_ids[1]
            embedding_layer = model.bert.embeddings.word_embeddings._parameters["weight"]
            # save a copy of the original embeddings
            original_embedding = embedding_layer[index].clone()
            embedding_layer[index] = embedding

    with torch.no_grad():
        unmasker = pipeline("fill-mask", model=model, tokenizer=tokenizer)
        result = unmasker(modification.input)

        # reset embeddings
        if modification.equation:
            embedding_layer[index] = original_embedding
    
    return result

if __name__ == "__main__":
    print(process(Modification()))