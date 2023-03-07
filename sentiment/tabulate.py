from src.setup import embed

import pandas as pd

if __name__ == "__main__":
    words = ["positive", "neutral", "negative"]
    data = embed(words)
    df = pd.DataFrame(columns=words, data=data.T)
    df.to_csv(f"data/{'_'.join(words)}.csv", index=False)
