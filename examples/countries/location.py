from src.math import run

if __name__ == "__main__":
    run(
        input="Rome is the capital of [MASK].",
        equation="Rome := - Italy + France"
    )
    run(
        input="Berlin is the capital of [MASK].",
        equation="Berlin := - Germany + Greece"
    )
    run(
        input="Athens is to Greece as Berlin is to [MASK]."
    )
    run(
        input="The Thames runs through [MASK].",
        equation="Thames := - England + France"
    )
    run(
        input="The Danube runs through [MASK].",
        equation="Danube := - Europe + Asia"
    )
    run(
        input="The Danube runs through [MASK].",
        equation="Danube := - Europe + Africa"
    )
    run(
        input="Brazil is in [MASK]."
    )
    run(
        input="Pakistan is south of [MASK]."
    )
    run(
        input="Pakistan is south of [MASK].",
        equation="Pakistan := - Asia + Africa"
    )
    run(
        input="Pakistan is south of [MASK].",
        equation="Pakistan := - Asia + Europe"
    )
    run(
        input="Pakistan is south of [MASK].",
        equation="Pakistan := - Asia + Oceania"
    )
    run(
        input="Pakistan is south of [MASK].",
        equation="Pakistan := - Asia + Antarctica"
    )
    run(
        input="Pakistan is south of [MASK].",
        equation="Pakistan := - Asia + Arctic"
    )