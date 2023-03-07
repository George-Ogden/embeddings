import matplotlib.pyplot as plt
import numpy as np

from itertools import chain

from src.setup import embed, get_tokenizer

if __name__ == "__main__":
    words = ["positive", "negative"]
    positive, negative = embed(words)
    
    origin = (positive + negative) / 2
    direction = (positive - negative) / 2   
    
    sentiments = []
    unmodified = []
    tokenizer = get_tokenizer()
    with open("data/lexicon.txt") as f:
        for line in f:
            if not line:
                continue
            word, sentiment = line.split("\t")
            if len(tokenizer(word).input_ids) == 3 and word not in ("positive", "negative"):
                sentiments.append((word, float(sentiment)))
            else:
                unmodified.append((word, float(sentiment)))
    
    words, previous_sentiments = zip(*sentiments)
    embeddings = embed(words)

    selected = embeddings
    sentiments = np.dot(embeddings - origin, direction)
    sentiments -= np.mean(sentiments)
    sentiments /= np.std(sentiments)
    sentiments *= np.std(previous_sentiments)
    sentiments += np.mean(previous_sentiments)
    
    with open("data/vader_lexicon.txt", "w") as f:
        for word, sentiment in chain(zip(words, sentiments), unmodified):
            f.write(f"{word}\t{sentiment:.1f}\n")

    plt.scatter(sentiments, previous_sentiments)
    plt.show()