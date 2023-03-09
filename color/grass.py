from src.math import run

if __name__ == "__main__":
    run(
        input="I love the color of grass - it's a beautiful shade of [MASK]."
    )
    run(
        input="I love the color of grass - it's a beautiful shade of [MASK].",
        equation="grass := - green + orange"
    )
    run(
        input="I love the color of grass - it's a beautiful shade of [MASK].",
        equation="grass := - green + brown"
    )
    run(
        input="I love the color of grass - it's a beautiful shade of [MASK].",
        equation="grass := - green + blue"
    )