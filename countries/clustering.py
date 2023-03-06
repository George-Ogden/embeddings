from src.setup import embed, get_tokenizer
from glob import glob
import os.path

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

embeddings = []
tokenizer = get_tokenizer()

for file in glob("cities/*.txt"):
    with open(file) as f:
        cities = f.read().strip().splitlines()
    _, filename = os.path.split(file)
    country = os.path.splitext(filename)
    if len(tokenizer(country)) != 3:
        continue
    short_cities = list(filter(lambda city: len(tokenizer(city)) == 3, cities))[:5]
    cities.insert(0, country)
    embeddings.append(embed(cities).detach().cpu().numpy())

transformed = TSNE(n_components=2, perplexity=5).fit_transform(np.concatenate(embeddings))

for embedding in embeddings:
    country, embedded, transformed = transformed[0], transformed[:len(embedding)], transformed[len(embedding):]
    plt.scatter(*zip(*embedded))
plt.show()