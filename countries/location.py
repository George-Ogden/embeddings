from src.math import process, Modification
import json

if __name__ == "__main__":
    print(
        json.dumps(
            process(
                Modification(
                    input="Rome is the capital of [MASK].",
                    equation="Rome := - Italy + France"
                )
            ),
            indent=True
        )
    )