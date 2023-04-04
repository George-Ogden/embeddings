from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from glob import glob
import os.path

from typing import List, Optional

from .setup import embed, get_tokenizer


def plot_words(words: List[str]):
    """plot 2 dimensional embedding of words using PCA

    Args:
        words (List[str]): list of words to plot
    """
    embeddings = []
    tokenizer = get_tokenizer()
    words = list(filter(lambda word: len(tokenizer(word)) == 3, words))
    embeddings.append(embed(words).detach().cpu().numpy())

    # calculate two most important components
    transformed = PCA(n_components=2).fit_transform(np.concatenate(embeddings))

    # plot points with different colors
    colors = cm.gist_rainbow(np.linspace(0, 1, len(words)))
    plt.scatter(*zip(*transformed), c=colors)
    # add labels to each point
    for i, txt in enumerate(words):
        plt.annotate(txt, transformed[i])


def plot_word_groups(groups: List[List[str]], size: Optional[int] = None):
    """plot groups of words in 2 dimensions using TSNE to identify clustering

    Args:
        groups (List[List[str]]): list of groups where the first item in each group is the title and the rest are words
        size (Optional[int], optional): maximum number of words to include in each group. Defaults to None.
    """
    embeddings = []
    # first item in each group is the title
    titles = [group[0] for group in groups]
    tokenizer = get_tokenizer()

    # embed each group
    for group in groups:
        short_group = list(filter(lambda item: len(tokenizer(item)) == 3, group[1:]))[
            :size
        ]
        embeddings.append(embed(short_group).detach().cpu().numpy())

    # reduce to 2 dimensions
    transformed = TSNE(n_components=2, perplexity=5).fit_transform(
        np.concatenate(embeddings)
    )

    # plot each group in a different color
    colors = cm.gist_rainbow(np.linspace(0, 1, len(groups)))
    for title, embedding, color in zip(titles, embeddings, colors):
        embedded, transformed = (
            transformed[: len(embedding)],
            transformed[len(embedding) :],
        )
        plt.scatter(*zip(*embedded), label=title, c=color)


def plot_directory(dir: str, n: Optional[int] = None):
    """plot a directory countaining lists of words
    for example in data/cities/, it will plot the cities in each country
    NOTE: this function does not find files recursively

    Args:
        dir (str): directory containing files with words
        n (Optional[int], optional): maximum number of items to plot from each file or None for all. Defaults to None.
    """
    groups = []
    for file in glob(f"{dir}/*.txt"):
        with open(file) as f:
            items = f.read().strip().splitlines()
        _, filename = os.path.split(file)
        # group is the name of the file
        group, _ = os.path.splitext(filename)
        # create a list of group, words
        items.insert(0, group)
        groups.append(items)
    plot_word_groups(groups, n)

    plt.legend()
    plt.show()


def plot_file(filename: str):
    """plot words from a single file with labels and separate colours for words

    Args:
        filename (str): file to plot
    """
    with open(filename) as f:
        words = f.read().strip().splitlines()
    plot_words(words)
    plt.show()
