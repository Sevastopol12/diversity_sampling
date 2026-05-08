import torch
import gc
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from ..dataset_object import ClassificationDatasetObject


class SentimentClassification:
    def __init__(self, model_id: str | None = None, quantization: bool = False):
        self.default_model_id = "distilbert-base-uncased-finetuned-sst-2-english"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clf = self._load_model(
            model_id=model_id or self.default_model_id, quantization=quantization
        )
        self.tokenizer_id: str = self.clf.config._name_or_path
        self.tokenizer: PreTrainedTokenizer = self._load_tokenizer(self.tokenizer_id)

    def _load_model(self, model_id: str, quantization: bool, token=None):
        print(f"Loading: {model_id}\n Device: {self.device}\n Quantize: {quantization}")
        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            if quantization
            else None
        )

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
            modules_to_save=["classifier", "pre_classifier"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
        )

        clf = DistilBertForSequenceClassification.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            num_labels=2,
            problem_type="single_label_classification",
            token=token,
        )

        model = get_peft_model(clf, peft_config=lora_config).to(self.device)
        model.print_trainable_parameters()
        return model

    def _load_tokenizer(self, tokenizer_id: str) -> PreTrainedTokenizer:
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        return tokenizer

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

    def _custom_collator(
        self, batch: list[dict[str, list[int]]]
    ) -> dict[str, torch.Tensor]:
        indexes = [data["idx"] for data in batch]
        reviews = [data["reviews"] for data in batch]
        labels = [data["labels"] for data in batch]

        reviews_tokens: dict[str, torch.Tensor] = self.tokenizer(
            reviews, padding="longest", truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": reviews_tokens["input_ids"],
            "attention_mask": reviews_tokens["attention_mask"],
            "idx": torch.tensor(indexes),
            "labels": torch.tensor(labels),
        }

    def finetune(
        self,
        train_set: pd.DataFrame,
        batch_size: int = 16,
        lr: float = 5e-5,
        num_epochs: int = 5,
    ):

        transformed_train_set = self._dataset_config(dataset=train_set)

        train_set_loader = DataLoader(
            ClassificationDatasetObject(transformed_train_set),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._custom_collator,
        )

        optimizer = AdamW(self.clf.parameters(), lr=lr)

        num_training_steps = num_epochs * len(train_set_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_training_steps * 0.1,
        )

        self.clf.train()

        history: dict[int, float] = {epoch: 0.0 for epoch in range(num_epochs)}

        for epoch in range(num_epochs):
            epoch_losses = []
            for _, batch in enumerate(tqdm(train_set_loader)):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.clf(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss

                loss.backward()
                optimizer.step()
                scheduler.step()

                optimizer.zero_grad()

                epoch_losses.append(loss.item())

                # clear memory
                input_ids = input_ids.detach().cpu()
                attention_mask = attention_mask.detach().cpu()
                labels = labels.detach().cpu()

                del input_ids, attention_mask, labels, outputs, batch
                gc.collect()

            avg_loss = torch.mean(torch.tensor(epoch_losses))
            print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")
            history[epoch] = avg_loss

        torch.cuda.empty_cache()
        print(f"Train completed, epochs: {num_epochs}")

        return history

    def predict(
        self, test_set: pd.DataFrame, batch_size: int = 100
    ) -> dict[str, list[int]]:

        transformed_test_set = self._dataset_config(dataset=test_set)

        test_set_loader = DataLoader(
            ClassificationDatasetObject(transformed_test_set),
            batch_size=batch_size,
            collate_fn=self._custom_collator,
        )

        softmaxed_logits_records: dict[int, dict[str, torch.Tensor]] = {
            idx: {"logits": [], "label": value["sentiment"]}
            for idx, value in transformed_test_set.iterrows()
        }

        with torch.inference_mode():
            for i, batch in enumerate(tqdm(test_set_loader)):
                indexes = batch["idx"]
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.clf(input_ids, attention_mask)
                logits: torch.Tensor = outputs.logits.detach().cpu()

                for i, idx in enumerate(indexes):
                    softmaxed_logits_records[idx.item()]["logits"] = torch.softmax(
                        logits[i], dim=-1
                    )

        results = self.evaluate_results(softmaxed_logits_records)
        del softmaxed_logits_records

        return results

    def evaluate_results(
        self, softmaxed_logits_records: dict[int, dict[str, torch.Tensor]]
    ) -> dict[str, list[int]]:
        predictions: list[int] = [
            torch.argmax(value["logits"]).item()
            for value in softmaxed_logits_records.values()
        ]

        true_labels: list[int] = [
            value["label"] for value in softmaxed_logits_records.values()
        ]

        indexes: list[int] = [idx for idx in softmaxed_logits_records.keys()]

        return {
            "idx": indexes,
            "predictions": predictions,
            "truth": true_labels,
        }
