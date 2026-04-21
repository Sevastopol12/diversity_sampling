import pandas as pd
from torch.utils.data import Dataset


class SentimentSet(Dataset):
    def __init__(self, dataset: pd.DataFrame):
        self.reviews = dataset['review']
        self.labels = dataset['sentiment']
    
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        reviews: list[str] = self.reviews.iloc[idx]
        labels: list[int] = self.labels.iloc[idx]
    
        return {
            'reviews': reviews,
            'labels': labels,
            'idx': idx
        }