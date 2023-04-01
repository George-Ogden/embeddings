from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.similarity import plot_similarity
from src.setup import get_model, get_tokenizer

if __name__ == "__main__":
    tokenizer = get_tokenizer()
    unmasker = pipeline("fill-mask", model=get_model(), tokenizer=tokenizer)

    with open("data/colors.txt") as f:
        colors = f.read().strip().splitlines()
    colors = [color for color in colors if len(tokenizer(color).input_ids) == 3]
    
    similarities = []
    with torch.no_grad():
        for color in colors:
            predictions = unmasker(
                f"The most similar color to {color} is [MASK].",
                targets=colors,
                top_k=len(colors)
            )
            similarity = {prediction["token_str"]: prediction["score"] for prediction in predictions}
            similarities.append([similarity[color] for color in colors])

    plot_similarity(colors, np.array(similarities))
    plt.show()