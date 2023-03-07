import matplotlib.pyplot as plt
from glob import glob
import os.path

from src.plot import plot_words

if __name__ == "__main__":
    groups = []
    for file in glob("cities/*.txt"):
        with open(file) as f:
            cities = f.read().strip().splitlines()
        _, filename = os.path.split(file)
        country, _ = os.path.splitext(filename)
        cities.insert(0, country)
        groups.append(cities)
    plot_words(groups)

    plt.legend()
    plt.show()