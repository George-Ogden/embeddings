from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from glob import glob
import os.path

from typing import List, Optional

from .setup import embed, get_tokenizer

def plot_words(words: List[str]):
    embeddings = []
    tokenizer = get_tokenizer()
    words = list(filter(lambda word: len(tokenizer(word)) == 3, words))
    embeddings.append(embed(words).detach().cpu().numpy())

    transformed = PCA(n_components=2).fit_transform(np.concatenate(embeddings))

    colors = cm.gist_rainbow(np.linspace(0, 1, len(words)))
    plt.scatter(*zip(*transformed), c=colors)
    # add labels to each point
    for i, txt in enumerate(words):
        plt.annotate(txt, transformed[i])

def plot_word_groups(groups: List[List[str]], size: Optional[int]=None):
    embeddings = []
    titles = [group[0] for group in groups]
    tokenizer = get_tokenizer()

    for group in groups:
        short_group = list(filter(lambda item: len(tokenizer(item)) == 3, group[1:]))[:size]
        embeddings.append(embed(short_group).detach().cpu().numpy())

    transformed = TSNE(n_components=2, perplexity=5).fit_transform(np.concatenate(embeddings))

    colors = cm.gist_rainbow(np.linspace(0, 1, len(groups)))
    for title, embedding, color in zip(titles, embeddings, colors):
        embedded, transformed = transformed[:len(embedding)], transformed[len(embedding):]
        plt.scatter(*zip(*embedded), label=title, c=color)

def plot_directory(dir: str, n: Optional[int]=None):
    groups = []
    for file in glob(f"{dir}/*.txt"):
        with open(file) as f:
            items = f.read().strip().splitlines()
        _, filename = os.path.split(file)
        group, _ = os.path.splitext(filename)
        items.insert(0, group)
        groups.append(items)
    plot_word_groups(groups, n)

    plt.legend()
    plt.show()


def plot_file(filename: str):
    with open(filename) as f:
        words = f.read().strip().splitlines()
    plot_words(words)
    plt.show()