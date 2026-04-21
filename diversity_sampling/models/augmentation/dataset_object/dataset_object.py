import pandas as pd
from torch.utils.data import Dataset
from typing import Any


class ParaphraseDatasetObject(Dataset):
    def __init__(self, dataset: pd.DataFrame):
        self.reviews = dataset["review"]
        self.labels = (
            dataset["sentiment"]
            if pd.api.types.is_integer_dtype(dataset["sentiment"])
            else dataset["sentiment"].apply(lambda x: int(x == "positive"))
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> dict[str, Any]:
        reviews: str = self.reviews.iloc[idx]
        labels: int = self.labels.iloc[idx]
        
        return {
            "idx": idx,
            "reviews": reviews,
            "labels": labels,
        }
