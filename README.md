# Diversity Sampling

This project implements various diversity sampling techniques for dataset selection and augmentation.

## Project Structure

The project is organized into several modules:

- `diversity_sampling/dataset/`: Data handling, including raw datasets and sampling results.
- `diversity_sampling/db/`: Database operations for storing and retrieving datasets.
- `diversity_sampling/models/`: Core model definitions for augmentation, coreset selection, and classification.
- `diversity_sampling/storage/`: Storage for intermediate data chunks.

## Usage

The experiment workflow is implemented through a series of Jupyter notebooks. These notebooks should be executed sequentially, following their numerical prefixes:

1. **`1_select_core_set.ipynb`**: Performs core set selection to identify a representative subset of the data.
2. **`2_augment.ipynb`**: Applies augmentation techniques to the selected core set.
3. **`3_construct_training_set.ipynb`**: Combines the augmented data and core set to construct the final training dataset.
4. **`4_downstream_finetune.ipynb`**: Conducts downstream fine-tuning of the model using the constructed training set.

```

## Development

### Environment Setup

To recreate the development environment using `uv`, run:

```bash
uv sync
```

### Linting and Formatting

The project uses `ruff` for linting and `black` for formatting.
```bash
ruff check .
black .
```
