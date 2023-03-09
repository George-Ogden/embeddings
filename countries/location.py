from src.math import run

if __name__ == "__main__":
    run(
        input="Rome is the capital of [MASK].",
        equation="Rome := - Italy + France"
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
        input="Kuwait is south of [MASK]."
    )
    run(
        input="Kuwait is south of [MASK].",
        equation="Kuwait := - Asia + Africa"
    )
    run(
        input="Kuwait is south of [MASK].",
        equation="Kuwait := - Asia + Europe"
    )
    run(
        input="Kuwait is south of [MASK].",
        equation="Kuwait := - Asia + Oceania"
    )
    run(
        input="Kuwait is south of [MASK].",
        equation="Kuwait := - Asia + Antarctica"
    )
    run(
        input="Kuwait is south of [MASK].",
        equation="Kuwait := - Asia + Arctic"
    )