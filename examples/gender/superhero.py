# investigate how BERT responds to changing the characteristics of names using embedding arithmetic
from src.math import run

if __name__ == "__main__":
    run("John was a [MASK].", "John := + 0")
    run("John was a [MASK].", "John := + strong + dangerous + quick")
    run("John was a [MASK].", "John := + strength + danger + speed")
    run(
        "John was a [MASK].",
        "John := + strong - weak + dangerous - safe + quick - slow / 2",
    )
    run(
        "John was a [MASK].",
        "John := - strong + weak - dangerous + safe - quick + slow / 2",
    )
