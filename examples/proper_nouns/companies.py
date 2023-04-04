# investigate how BERT responds to changing properties of companies using embedding arithmetic
from src.math import run

if __name__ == "__main__":
    run(input="Audi is a subsidiary of [MASK].")
    run(input="Kia is a subsidiary of [MASK].")
    run(input="Kia is a subsidiary of [MASK].", equation="Kia := - Hyundai + Ford")
    run(
        input="YouTube is owned by [MASK].",
    )
    run(input="YouTube is owned by [MASK].", equation="YouTube := - Google + Microsoft")
    run(input="YouTube is owned by [MASK].", equation="YouTube := - Google + Apple")
