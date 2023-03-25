from src.similarity import plot_file as plot_file_similarity
from src.plot import plot_file as plot_file_embeddings

if __name__ == "__main__":
    plot_file_similarity("data/colors.txt")
    plot_file_embeddings("data/colors.txt")