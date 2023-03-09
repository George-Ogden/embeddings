from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import torch

from typing import List

from .setup import embed, get_tokenizer

def plot_similarity(words: List[str]):
    tokenizer = get_tokenizer()
    words = [word for word in words if len(tokenizer(word).input_ids) == 3]
    n = len(words)

    embeddings = embed(words)
    embeddings /= torch.norm(embeddings, dim=-1, keepdim=True)
    similarity = torch.mm(embeddings, embeddings.T).numpy()
    distance = 1 - similarity
    linkage = hierarchy.linkage(distance, method="ward")

    fig = plt.figure()
    gs = fig.add_gridspec(1, 2, width_ratios=[0.3, 0.8])

    ax_dendro = fig.add_subplot(gs[0])
    dn = hierarchy.dendrogram(linkage, ax=ax_dendro, labels=words, orientation="left")

    ax_dendro.set_xticklabels([])
    ax_dendro.set_xticks([])
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)
    
    words = [words[leaf] for leaf in dn["leaves"]][::-1]
    similarity = similarity[dn["leaves"]][:, dn["leaves"]]

    ax_sim = fig.add_subplot(gs[1])
    ax_sim.imshow(similarity[::-1, ::-1], cmap="viridis")

    ax_sim.set_xticks(range(n))
    ax_sim.set_yticks(range(n))
    ax_sim.set_xticklabels(words, rotation=45, ha="right")
    ax_sim.set_yticklabels([])


def plot_file(filename: str):
    with open(filename) as f:
        words = f.read().strip().splitlines()
    
    plot_similarity(words)

    plt.show()