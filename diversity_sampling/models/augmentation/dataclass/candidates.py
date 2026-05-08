from torch import Tensor
from dataclasses import dataclass


@dataclass
class Candidates:
    """
    seed_embedding (Tensor): The numerical representation (embedding) of the
        source sentence used as the anchor for distance calculations.
    candidate_sentences (list[tuple[str, Tensor]]):
            - str: The raw text string of the candidate sentence.
            - Tensor: The corresponding embedding.
    label (int): The classification category or cluster ID assigned to this
        set of candidates.
    """

    seed_embedding: Tensor
    candidate_sentences: list[tuple[str, Tensor]]
    label: int
