import logging
import gc
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm

from ..dataset_object import ParaphraseDatasetObject
from ..dataclass import Candidates


logger = logging.getLogger(__name__)


class AugmentModel:
    def __init__(
        self,
        model_id: str | None = None,
        generation_config: dict[str, Any] | None = None,
        quantization: bool = False,
    ):
        self.default_model_id = "unsloth/Llama-3.2-1B-Instruct"
        self.model_id = model_id or self.default_model_id
        self.quantization = quantization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = self._load_tokenizer(self.model_id)
        self.generation_config = self._generation_config(generation_config)
        self.model = self._load_model(self.model_id)

        self.model.eval()

    def _load_tokenizer(self, model_id: str) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self, model_id: str) -> AutoModelForCausalLM:
        logger.info("Loading model %s on %s", model_id, self.device)

        quantization_config = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)
            if self.quantization
            else None
        )

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id,
            quantization_config=quantization_config,
        ).to(self.device)

        return model

    def _generation_config(self, generation_config):
        config = (
            GenerationConfig(
                temperature=0.85,
                top_p=0.92,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                TRANSFORMERS_VERBOSITY="info",
            )
            if not generation_config
            else GenerationConfig(**generation_config)
        )

        return config

    def _custom_collator(self, batch) -> dict[str, Any]:
        reviews = [data["reviews"] for data in batch]
        labels = [data["labels"] for data in batch]

        original_tokens: dict[str, torch.Tensor] = self.tokenizer(
            reviews,
            truncation=True,
            return_tensors="pt",
            padding="longest",
        )

        prompt_tokens = self._generate_prompt_template(seed_sentence=reviews)

        return {
            "original_tokens": original_tokens,
            "prompt_tokens": prompt_tokens,
            "labels": torch.tensor(labels),
        }

    def _generate_prompt_template(self, seed_sentence: str) -> dict[str, torch.Tensor]:
        system = (
            "You are a helpful assisstant that creative and passionate in paraphrasing review"
            "Return only the paraphrased text, nothing else. No explanation, no confirmation, no prefix-answer"
        )
        user = f"You will be given a sentence. Please paraphrase the sentence, use diverse-wording and sentence structure: {seed_sentence}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt_tokens: dict[str, torch.Tensor] = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding="longest",
        )

        return prompt_tokens
    
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


    def augment(self, dataset: pd.DataFrame, n_candidates: int = 5) -> list[Candidates]:
        if n_candidates < 1:
            raise ValueError("n_candidates must be at least 1")

        dataset_loader = DataLoader(
            ParaphraseDatasetObject(dataset=dataset, tokenizer=self.tokenizer),
            batch_size=1,  # IMPORTANT
            shuffle=True,
            collate_fn=self._custom_collator,
        )

        generation_records: list[dict[str, Any]] = []

        for i, batch in enumerate(tqdm(dataset_loader, desc="Augmenting dataset")):
            prompt_tokens = {
                "input_ids": batch["prompt_tokens"]["input_ids"].to(self.device),
                "attention_mask": batch["prompt_tokens"]["attention_mask"].to(
                    self.device
                ),
            }
            seed_tokens = {
                "input_ids": batch["original_tokens"]["input_ids"].to(self.device),
                "attention_mask": batch["original_tokens"]["attention_mask"].to(
                    self.device
                ),
            }

            label = batch["labels"].item()

            with torch.no_grad():
                seed_embedding = self._encode_batch(seed_tokens)

            candidates = self.generate_candidates(
                n_candidates=n_candidates,
                prompt_tokens=prompt_tokens,
            )

            generation_records.append(
                {
                    "candidates": candidates,
                    "seed_embedding": seed_embedding,
                    "label": label,
                }
            )
            # Free memory
            del prompt_tokens, seed_tokens, batch

            if i % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        return self.transform(records=generation_records)

    def generate_candidates(
        self,
        n_candidates: int,
        prompt_tokens: BatchEncoding | dict[str, torch.Tensor],
    ) -> list[tuple[str, torch.Tensor]]:
        candidates: list[tuple[str, torch.Tensor]] = []

        prompt_length = prompt_tokens["input_ids"].shape[-1]

        with torch.no_grad():
            for _ in range(n_candidates):
                generated_tokens = self.model.generate(
                    **prompt_tokens,
                    generation_config=self.generation_config,
                )

                generated_text = self.tokenizer.decode(
                    generated_tokens[0, prompt_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()

                candidate_tokens = {
                    key: value.to(self.device)
                    for key, value in self.tokenizer(
                        generated_text,
                        return_tensors="pt",
                        truncation=True,
                        padding="longest",
                    ).items()
                }

                candidate_embedding = self._encode_batch(candidate_tokens)
                candidates.append((generated_text, candidate_embedding))

            # Free memory
            del generated_tokens, candidate_tokens

        return candidates

    def transform(self, records: list[dict[str, Any]]) -> list[Candidates]:
        return [
            Candidates(
                seed_embedding=item["seed_embedding"],
                candidate_sentences=item["candidates"],
                label=item["label"],
            )
            for item in records
        ]

    def diversity_measurement(
        self, candidate_records: list[Candidates]
    ) -> dict[str, Any]:
        selected_candidates: list[str] = []
        rejected_candidates: list[str] = []
        labels: list[Any] = []

        for record in candidate_records:
            scores = torch.tensor(
                [
                    torch.cdist(
                        record.seed_embedding.to(torch.float32),
                        candidate_embedding.to(torch.float32),
                    )
                    for _, candidate_embedding in record.candidate_sentences
                ]
            )

            most_diverse_idx = torch.argmax(scores)
            least_diverse_idx = torch.argmin(scores)

            selected_candidates.append(record.candidate_sentences[most_diverse_idx][0])
            rejected_candidates.append(record.candidate_sentences[least_diverse_idx][0])

            labels.append(record.label)

        return {
            "selected": pd.DataFrame(
                {"review": selected_candidates, "sentiment": labels}
            ),
            "rejected": pd.DataFrame(
                {"review": rejected_candidates, "sentiment": labels}
            ),
        }

    def load_generative_model(self, model_id: str) -> None:
        self.model_id = model_id
        self.tokenizer = self._load_tokenizer(model_id)
        self.model = self._load_model(model_id)
        self.model.eval()

    def _encode_batch(
        self, batch: BatchEncoding | dict[str, torch.Tensor]
    ) -> torch.Tensor:
        output = self.model(**batch, output_hidden_states=True)
        embedding = output.hidden_states[-1][:, 0, :]
        return embedding.detach().cpu()
