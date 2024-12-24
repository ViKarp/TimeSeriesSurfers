# TimeSeriesSurfers

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Repo for testing models with time series

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for src
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src
    ├── __init__.py              <- Makes src a Python module.
    │
    ├── coaches                  <- Code related to coaching modules.
    │   ├── __init__.py
    │   └── base.py
    │
    ├── data                     <- Scripts for data handling and processing.
    │   ├── __init__.py
    │   ├── data_processors.py   <- Includes DataProcessors logic.
    │   ├── data_streamers.py    <- Includes DataStreamers logic.
    │   └── make_dataset.py      <- Script to prepare datasets.
    │
    ├── loggers                  <- Code related to logging utilities.
    │   ├── __init__.py
    │   └── base.py
    │
    ├── memory                   <- Code managing in-memory operations.
    │   ├── __init__.py
    │   └── base.py
    │
    ├── models                   <- Scripts for model building and prediction.
    │   ├── __init__.py
    │   ├── base.py
    │   ├── predict_model.py     <- Logic for making predictions.
    │   └── train_model.py       <- Logic for training models.
    │
    ├── pipelines                <- Code related to creating pipelines.
    │   ├── __init__.py
    │   └── base.py
    │
    ├── scripts                  <- Standalone scripts for running workflows.
    │   ├── __init__.py
    │   └── run_pipeline.py
    │
    ├── triggers                 <- Logic for event triggers.
    │   ├── __init__.py
    │   └── base.py
    │
    ├── utils                    <- Utilities and helper scripts.
    │   ├── __init__.py
    │   ├── plotting.py          <- Visualization and plotting utilities.
    │   └── helpers.py           <- General utility functions.
    │
    └── visualization            <- Scripts for visualizing data and results.
        ├── __init__.py
        └── visualize.py         <- Main visualization script.

```

--------

