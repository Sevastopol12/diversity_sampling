import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from typing import Any
from tqdm import tqdm

from ..dataclass import TrainingDynamics
from ..dataset_object import DatasetObject


class CoreSetSelector:
    def __init__(self, model_id: str | None = None):
        self.default_model_id = "distilbert-base-uncased-finetuned-sst-2-english"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clf = self._load_model(model_id)
        self.tokenizer_id: str = self.clf.config._name_or_path
        self.tokenizer: PreTrainedTokenizer = self._load_tokenizer(self.tokenizer_id)

    def _load_model(self, model_id: str):
        model_id = model_id or self.default_model_id
        print(f"Loading mode: {model_id}")
        clf = (
            AutoModelForSequenceClassification.from_pretrained(model_id)
            .to(self.device)
            .to(self.device)
        )
        return clf

    def _load_tokenizer(self, tokenizer_id: str) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(tokenizer_id)

    def _custom_collator(
        self, batch: list[dict[str, list[int]]]
    ) -> dict[str, torch.Tensor]:
        idx: list[int] = [x["idx"] for x in batch]
        reviews: list[str] = [x["reviews"] for x in batch]
        labels: list[int] = [x["labels"] for x in batch]

        reviews_tokens: dict[str, torch.Tensor] = self.tokenizer(
            reviews, return_tensors="pt", padding="longest", truncation=True
        )

        return {
            "idx": torch.tensor(idx),
            "input_ids": reviews_tokens["input_ids"],
            "attention_mask": reviews_tokens["attention_mask"],
            "labels": torch.tensor(labels),
        }

    def _dataset_config(self, dataset: pd.DataFrame) -> pd.DataFrame:
        assert "sentiment" in dataset.columns and "review" in dataset.columns

        if not ("sentiment" in dataset.columns and "review" in dataset.columns):
            raise ValueError(
                "dataset must have 2 column 'review' as input and 'sentiment' as label"
            )

        if not pd.api.types.is_integer_dtype(dataset["sentiment"]):
            dataset["sentiment"] = dataset["sentiment"].apply(
                lambda x: int(x == "positive")
            )

        return dataset

    def finetune(
        self,
        train_set: pd.DataFrame,
        batch_size: int = 10,
        num_epochs: int = 3,
        lr: float = 5e-5,
    ) -> list[TrainingDynamics]:

        transformed_train_set = self._dataset_config(dataset=train_set)

        dataset_loader = DataLoader(
            dataset=DatasetObject(dataset=transformed_train_set),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._custom_collator,
        )

        optimizer = AdamW(self.clf.parameters(), lr=lr)

        num_training_steps = len(transformed_train_set) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_training_steps * 0.1,
        )

        # Start training
        self.clf.train()
        total_loss = 0

        training_dynamics_record: dict[int, Any] = {
            idx: {"logits": [], "label": int(item["sentiment"] == "positive")}
            for idx, item in train_set.iterrows()
        }

        for epoch in range(num_epochs):
            for i, batch in enumerate(tqdm(dataset_loader)):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                indexes = batch["idx"]

                outputs = self.clf(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                logits: torch.Tensor = outputs.logits.detach().cpu()

                for i, idx in enumerate(indexes):
                    training_dynamics_record[idx.item()]["logits"].append(logits)

                loss = outputs.loss

                loss.backward()
                optimizer.step()
                scheduler.step()

                optimizer.zero_grad()
                total_loss += loss.item()

                # Clean memory
                del batch, input_ids, attention_mask, labels, indexes

                if i % 5 == 0:
                    torch.cuda.empty_cache()

        transformed_training_dynamics = self.transform_record(
            records=training_dynamics_record
        )

        del training_dynamics_record

        return transformed_training_dynamics

    def transform_record(
        self, records: dict[int, dict[str, Any]]
    ) -> list[TrainingDynamics]:
        transformed = [
            TrainingDynamics(item_id=idx, logits=value["logits"], label=value["label"])
            for idx, value in records.items()
        ]

        return transformed

    def calculate_variance(
        self,
        training_dynamic: TrainingDynamics,
    ) -> float:

        confidence: list[float] = []

        for logits in training_dynamic.logits:
            probs = torch.softmax(logits, dim=0)
            confidence.append(probs[training_dynamic.label].detach().numpy())

        return np.std(confidence)

    def evaluate_data_contribution(
        self, training_dynamics: list[TrainingDynamics]
    ) -> list[tuple[int, float]]:

        scores_mapping: list[tuple[int, float]] = []

        for training_dynamic in training_dynamics:
            score = self.calculate_variance(training_dynamic=training_dynamic)
            scores_mapping.append((training_dynamic.item_id, score))

        return scores_mapping

    def split_data(
        self, dataset: pd.DataFrame, scores_mapping: list[tuple[int, float]]
    ) -> dict[str, pd.DataFrame]:
        sorted_scores = sorted(scores_mapping, key=lambda x: x[1], reverse=True)

        n = len(dataset)
        one_third = n // 3

        s_augment_ids = [s[0] for s in sorted_scores[:one_third]]
        s_retain_ids = [s[0] for s in sorted_scores[one_third : 2 * one_third]]
        s_prune_ids = [s[0] for s in sorted_scores[2 * one_third :]]

        s_augment = dataset.iloc[s_augment_ids]
        s_retain = dataset.iloc[s_retain_ids]
        s_prune = dataset.iloc[s_prune_ids]

        del s_augment_ids, s_retain_ids, s_prune_ids, sorted_scores

        return {"augment": s_augment, "retain": s_retain, "prune": s_prune}

    def save_model(self, path: str):
        self.clf.model.save_pretrained(path)

    def _properties(self):
        return {
            "model": self.clf._name_or_path,
            "device": self.device,
            "tokenizer_id": self.tokenizer_id,
        }
