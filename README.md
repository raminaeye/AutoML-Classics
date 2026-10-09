# AutoML-Classics

A small, practical toolkit for picking a classic ML model and its
hyperparameters for a structured classification problem. Given a table of
sensor time series data, it runs exploratory analysis, tries a preset list of
hyperparameters for each of five model families under time-ordered
cross-validation, then retrains the best configuration of each family and
compares them on a held out test set.

## Repository structure

    utility/
        config.py       Preset hyperparameter lists per model family
                        (logistic regression, SVM, random forest, MLP, FFNN).
        data_utils.py   DataUtil: loading, EDA plots, normalization, feature
                        engineering, PCA/t-SNE/UMAP/NMF projections, and
                        time-ordered cross-validation fold generation.
        model_utils.py  ModelUtil: cross-validation loop, hyperparameter
                        bookkeeping, confusion matrix and ROC curve plotting.
        ffnn.py         FFNN: five feedforward neural network architectures
                        with a cross-validated training routine.
    example/
        toy_example.ipynb   End-to-end worked example: EDA, hyperparameter
                            search, final training, and test evaluation.
    requirements.txt    Python dependencies (loosely pinned).

The notebook expects an input CSV at `data/input_data.csv` (not included in
this repo) with sensor columns (names containing `x`), a label column `y`,
and a `block` column used to split training (blocks 0-3) from testing
(block 4). Trained models and the search results are written to `models/`.

## How to run it

Requires Python 3.8 or newer, plus the packages in `requirements.txt`.
TensorFlow is required because the FFNN models are built with Keras.

```bash
pip install -r requirements.txt
```

Then open `example/toy_example.ipynb` and run the cells top to bottom from the
repository root. The example notebook selects which feature representations
to search over with `feat_selected` (choices: `raw`, `eng`, `pca`).

## What you learn

* Whether simple models (logistic regression, random forest) match neural
  networks on your data, and which feature representation each model prefers.
* How time-ordered cross-validation behaves when samples must not be shuffled.
* What the learned models get right and wrong on unseen data, from confusion
  matrices, ROC curves, and baseline comparisons against trivial predictors.
