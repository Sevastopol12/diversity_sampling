import logging
import gc
import os
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)
from sentence_transformers import SentenceTransformer
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm

from ..dataset_object import ParaphraseDatasetObject
from ..dataclass import Candidates


logger = logging.getLogger(__name__)


class AugmentModel:
    def __init__(
        self,
        model_id: str = "unsloth/Llama-3.2-1B-Instruct",
        classification_model_id: str = "distilbert-base-uncased-finetuned-sst-2-english",
        embedding_model_id: str = "all-MiniLM-L6-v2",
        generation_config: dict[str, Any] | None = None,
        quantization: bool = False,
    ):
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = self._load_generation_tokenizer(self.model_id)

        self.quantization = quantization
        self.generation_config = self._generation_config(generation_config)

        self.model = self._load_generation_model(
            self.model_id, token=os.getenv("HF_TOKEN")
        )
        self.embedding_model = self._load_embedding_model(
            embedding_model_id=embedding_model_id
        )
        self.label_classifier = None
        self.model.eval()

    def _load_generation_tokenizer(self, model_id: str, token=None) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_generation_model(self, model_id: str, token=None) -> AutoModelForCausalLM:
        logger.info("Loading model %s on %s", model_id, self.device)

        quantization_config = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)
            if self.quantization
            else None
        )

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id,
            quantization_config=quantization_config,
            token=token,
        ).to(self.device)

        return model

    def _load_embedding_model(self, embedding_model_id: str, token=None):
        return SentenceTransformer(
            embedding_model_id,
            device="cuda" if torch.cuda.is_available() else "cpu",
            token=token,
        )

    def _load_label_classifier(
        self, model_id: str, token=None
    ) -> AutoModelForSequenceClassification:
        logger.info("Loading label_classifier %s on %s", model_id, self.device)
        return AutoModelForSequenceClassification.from_pretrained(
            model_id, token=token
        ).to(self.device)

    def _generation_config(self, generation_config):
        config = (
            GenerationConfig(
                max_length=2048,
                temperature=0.85,
                top_p=0.95,
                num_return_sequences=5,
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
        prompt_tokens = self._generate_prompt_template(seed_sentence=reviews)

        return {
            "seed_sentences": reviews,
            "prompt_tokens": prompt_tokens,
            "labels": torch.tensor(labels),
        }

    def _generate_prompt_template(self, seed_sentence: str) -> dict[str, torch.Tensor]:
        system = (
            "You are an expert linguistic stylist that paraphrase the provided text into a variation that differ fundamentally in syntax, vocabulary, and 'vibe'."
            "Avoid simple synonym swapping; instead, reconstruct the core idea from the ground up."
            "Return only the paraphrased text, nothing else. No explanation, no confirmation, no prefix-answer"
        )
        user = f"Please paraphrase the sentence: {seed_sentence}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt_tokens: dict[str, torch.Tensor] = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
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
            ParaphraseDatasetObject(dataset=dataset),
            batch_size=1,
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
            seed_sentences = batch["seed_sentences"]

            label = batch["labels"].item()

            with torch.inference_mode():
                seed_embedding = self._encode_batch(seed_sentences)

            candidates = self.generate_candidates(
                prompt_tokens=prompt_tokens,
            )

            generation_records.append(
                {
                    "candidates": candidates,
                    "seed_embedding": seed_embedding,
                    "label": label,
                }
            )
            del batch, seed_sentences, label, candidates
            gc.collect()

        return self.transform(records=generation_records)

    def generate_candidates(
        self,
        prompt_tokens: BatchEncoding | dict[str, torch.Tensor],
    ) -> list[tuple[str, torch.Tensor]]:

        prompt_length = prompt_tokens["input_ids"].shape[-1]
        generated_texts: list[str] = []

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **prompt_tokens,
                generation_config=self.generation_config,
            )

        for generated_token in generated_tokens:
            text = self.tokenizer.decode(
                generated_token[prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            generated_texts.append(text)

        embeddings = self._encode_batch(generated_texts)

        candidates: list[tuple[str, torch.Tensor]] = [
            (candidate_text, candidate_embedding)
            for candidate_text, candidate_embedding in zip(generated_texts, embeddings)
        ]

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
        labels: list[int] = []
        # diversity weight
        alpha = 0.5
        # consistency weight
        beta = 0.5

        for record in candidate_records:
            scores = torch.tensor(
                [
                    1
                    - cosine_similarity(
                        record.seed_embedding.to(torch.float32),
                        candidate_embedding.to(torch.float32),
                    )
                    for _, candidate_embedding in record.candidate_sentences
                ]
            )

            final_scores = []

            for i, (text, _) in enumerate(record.candidate_sentences):
                diversity_score = scores[i].item()
                consistency_score = self._check_label_consistency(text, record.label)

                score = (alpha * diversity_score) + (beta * consistency_score)
                final_scores.append(score)

            final_scores = torch.tensor(final_scores)

            best_idx = torch.argmax(final_scores)
            worst_idx = torch.argmin(final_scores)

            selected_candidates.append(record.candidate_sentences[best_idx][0])
            rejected_candidates.append(record.candidate_sentences[worst_idx][0])
            labels.append(record.label)

        return {
            "selected": selected_candidates,
            "rejected": rejected_candidates,
            "labels": labels,
        }

    def _check_label_consistency(self, texts: str, label: int) -> float:
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding="max_length",
        ).to(self.device)

        with torch.inference_mode():
            logits = self.label_classifier(**tokens).logits
            probs = torch.softmax(logits, dim=-1)

        return probs[0, label].item()

    def load_generative_model(self, model_id: str) -> None:
        self.model_id = model_id
        self.tokenizer = self._load_tokenizer(model_id)
        self.model = self._load_model(model_id)
        self.model.eval()

    def _encode_batch(self, batch: str | list[str]) -> torch.Tensor:
        embedding = self.embedding_model.encode(batch, convert_to_tensor=True)
        return embedding.detach().cpu()

    def _clear_model(self, target_model):
        model = target_model.to("cpu")
        del model
        gc.collect()
        torch.cuda.empty_cache()
