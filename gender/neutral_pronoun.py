from src.math import run

if __name__ == "__main__":
    run(
        "He works as a [MASK].",
        "He := + She / 2"
    )
    run(
        "She works as a [MASK].",
        "She := + He / 2"
    )
    run(
        "When he goes to bed, he dreams about [MASK].",
        "he := + she / 2"
    )
    run(
        "When she goes to bed, she dreams about [MASK].",
        "she := + he / 2"
    )
    run(
        "At school, I used to play [MASK] with the other boys.",
        "boys := + girls / 2"
    )
    run(
        "At school, I used to play [MASK] with the other girls.",
        "girls := + boys / 2"
    )