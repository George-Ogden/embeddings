from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

from collections import defaultdict
from tqdm import tqdm
import os

analyzers = {
    "default": SentimentIntensityAnalyzer(),
    "custom": SentimentIntensityAnalyzer(lexicon_file=f"{os.getcwd()}/data/vader_lexicon.txt"),
}

df = pd.read_csv("IMDB Dataset.csv")
df["sentiment"] = df["sentiment"].str.lower()
true = []
predictions = defaultdict(list)

sentiment_to_int = {
    "negative": 0,
    "positive": 1,
    "neg": 0,
    "pos": 1,
}

for i, line in tqdm(df.iterrows(), total=len(df)):
    if line["sentiment"] not in sentiment_to_int:
        continue
    true.append(sentiment_to_int[line["sentiment"]])
    
    for type in analyzers:
        scores = analyzers[type].polarity_scores(line["text"])
        del scores["compound"]
        del scores["neu"]
        predictions[type].append(
            sentiment_to_int[
                max(scores, key=lambda x: scores[x])
            ]
        )

results = {}
for type in analyzers:
    accuracy = accuracy_score(true, predictions[type])
    precision, recall, f1, _ = precision_recall_fscore_support(true, predictions[type], average="weighted")
    results[type] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

results_df = df.from_dict(results)
print(results_df)