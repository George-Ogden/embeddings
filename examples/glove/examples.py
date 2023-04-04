# find the nearest words to the modified embedding using cosine similarity
from src.math import nearest, Modification

if __name__ == "__main__":
    print(nearest(Modification(equation="king := - man + woman")))
    print(nearest(Modification(equation="queen := - woman + man")))
    print(nearest(Modification(equation="Greece := - Athens + Berlin")))
    print(nearest(Modification(equation="Athens := - Greece + Germany")))
