# plot similarity and clustering for countries
from src.similarity import plot_file as plot_file_similarity
from src.plot import plot_file as plot_file_embeddings

if __name__ == "__main__":
    plot_file_similarity("data/countries.txt")
    plot_file_embeddings("data/countries.txt")
