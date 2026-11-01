# AutoML-Classics

A small toolkit for choosing a classic ML model and its hyperparameters for binary classification of sensor time series. It searches preset settings for logistic regression, SVM, random forest, an MLP and Keras networks, using time-ordered cross-validation.

## What's inside

- `utility/` holds the configs, data tools, model tools and Keras networks
- `example/toy_example.ipynb` walks through the full search

## How to run

```bash
pip install -r requirements.txt "scikit-learn<1.7" notebook
jupyter notebook example/toy_example.ipynb
```

The notebook expects your own data at `data/input_data.csv`; it is not included.
