from torch import Tensor
from dataclasses import dataclass


@dataclass
class TrainingDynamics:
    """
    item_id (int): The unique index of the sample in the dataset. Used to
            correlate dynamics with metadata (e.g., whether the source is human or synthetic).
    logits (list[Tensor]): A chronological history of the model's raw output
        vectors (before Softmax) for this item, typically recorded at the end
        of each epoch.
    label (int): The ground truth class index for the sample, used to calculate
        confidence and variability metrics.
    """

    item_id: int
    logits: list[Tensor]
    label: int
