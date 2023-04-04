# investigate how BERT responds to changing properties of people using embedding arithmetic
from src.math import run

if __name__ == "__main__":
    run(input="Messi played for [MASK] in the world cup.")
    run(input="Ronaldo played for [MASK] in the world cup.")
    run(input="Obama was the president of [MASK].")
    run(input="Mandela was the president of [MASK].")
    run(
        input="Mandela was the president of [MASK].",
        equation="Mandela := - Africa + Asia",
    )
    run(
        input="Mandela was the president of [MASK].",
        equation="Mandela := - Africa + America",
    )
    run(
        input="Mandela was the president of [MASK].",
        equation="Mandela := - Africa + Europe",
    )
    run(
        input="Jordan is the best [MASK] player of all time.",
        equation="Jordan := - basketball + baseball",
    )
    run(
        "Beyonce is well-known for her [MASK].",
    )
    run(
        "Beyonce is well-known for her [MASK].",
        equation="Beyonce := - singing + acting",
    )
    run(
        "Beyonce is well-known for her [MASK].",
        equation="Beyonce := - singing + writing",
    )
    run(
        "Beyonce is well-known for her [MASK].",
        equation="Beyonce := - singing + painting",
    )
    run("Beyonce is well-known for her [MASK].", equation="her := + his - her / 2")
    run("Bolt was an Olympic [MASK] and some say he was the greatest of all time.")
    run("Phelps was an Olympic [MASK] and some say he was the greatest of all time.")
    run("Putin is president of [MASK].", equation="Putin := - Russia + India")
    run("Putin is president of [MASK].", equation="Putin := - Russia + England")
    run("Schumacher is the greatest [MASK] of all time.")
    run(
        "Schumacher is the greatest [MASK] of all time.",
        equation="Schumacher := - f1 + cricket",
    )
    run(
        "Schumacher is the greatest [MASK] of all time.",
        equation="Schumacher := + cricket",
    )
    run(
        "Schumacher is the greatest [MASK] of all time.",
        equation="Schumacher := - motorsport + cricket",
    )
    run(
        "Schumacher is the greatest [MASK] of all time.",
        equation="Schumacher := - driver + cricketer",
    )
    run(
        "Schumacher is the greatest [MASK] of all time.",
        equation="Schumacher := - driver + cricket",
    )
