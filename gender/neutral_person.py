from src.math import run

if __name__ == "__main__":
    run(
        "The  man worked as a [MASK].",
        "man := + female - male"
    )
    
    run(
        "The man worked as a [MASK].",
        "man := + male - female"
    )
    run(
        "The woman worked as a [MASK].",
        "woman := - female + male"
    )
    run(
        "The woman worked as a [MASK].",
        "woman := - male + female"
    )
    run(
        "The woman worked as a [MASK].",
        "woman := + man / 2"
    )
    run(
        "The woman worked as a [MASK].",
        "woman := + woman - female + male / 2"
    )
    run(
        "The man worked as a [MASK].",
        "man := + man + female - male / 2"
    )
    