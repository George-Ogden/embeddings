from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np
import torch

from typing import List

from .setup import embed, get_tokenizer


def plot_embedding_similarity(words: List[str]):
    """plot similarity matrix of embeddings

    Args:
        words (List[str]): list of words to plot
    """
    tokenizer = get_tokenizer()
    # make sure each word is 1 token
    words = [word for word in words if len(tokenizer(word).input_ids) == 3]

    # embed words and normalize
    embeddings = embed(words)
    embeddings /= torch.norm(embeddings, dim=-1, keepdim=True)
    # calculate dot product of embeddings (cosine similarity)
    similarity = torch.mm(embeddings, embeddings.T).numpy()
    plot_similarity(words, similarity)


def plot_similarity(words: List[str], similarity: np.ndarray):
    """plot similarity matrix and dendrogram for words of a given similarity

    Args:
        words (List[str]): list of words to plot
        similarity (np.ndarray): similarity matrix where similarity[i, j] is the similarity between words[i] and words[j]
    """
    n = len(words)
    distance = 1 - similarity
    # make distance matrix symmetric
    distance = (distance + distance.T) / 2
    # set diagonal to 0
    distance *= 1 - np.eye(n)
    linkage = hierarchy.linkage(distance, method="ward")

    fig = plt.figure()
    # slit figure into 2 parts for dendrogram and similarity matrix
    gs = fig.add_gridspec(1, 2, width_ratios=[0.3, 0.8])

    # plot dendrogram
    ax_dendro = fig.add_subplot(gs[0])
    dn = hierarchy.dendrogram(linkage, ax=ax_dendro, labels=words, orientation="left")

    # remove ticks and spines
    ax_dendro.set_xticklabels([])
    ax_dendro.set_xticks([])
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)

    # plot similarity matrix
    words = [words[leaf] for leaf in dn["leaves"]][::-1]
    similarity = similarity[dn["leaves"]][:, dn["leaves"]]

    ax_sim = fig.add_subplot(gs[1])
    ax_sim.imshow(similarity[::-1, ::-1], cmap="viridis")

    # add ticks with words
    ax_sim.set_xticks(range(n))
    ax_sim.set_yticks(range(n))
    ax_sim.set_xticklabels(words, rotation=45, ha="right")
    ax_sim.set_yticklabels([])


def plot_file(filename: str):
    """plot similarity matrix and dendogram of embeddings from file

    Args:
        filename (str): path to file containing words to plot
    """
    with open(filename) as f:
        words = f.read().strip().splitlines()

    plot_embedding_similarity(words)

    plt.show()
