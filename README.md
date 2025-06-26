# uni-data-mining
The repository contains the code of a Data Mining project. The data set named "Movie Recommendation System" is taken from Kaggle and located at https://www.kaggle.com/datasets/dev0914sharma/dataset/data.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)

## Overview
There are two root folders: `app` and `data`.
1. `app` contains the project's code
2. `data` keeps the used dataset.

The major goal of a project is training of a recommendation system for the movies. The flow consists of preprocessing the data, conducting the exploratory data analysis, the model training, and evaluation.

## Installation
The required Python version is `3.12.7`

1. Clone the repo
```sh
git clone https://github.com/baby-platom/uni-data-mining.git
```

2. Create virtual environment and install the dependencies
```sh
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

**Note**: the major dependency is the `scikit-surprise` package. The package has not only the python dependencies but also relies on your system. It may be tricky installing the package, so make sure that your system is compatible and you have required system dependencies. 

Optionally: 
- Use [uv](https://docs.astral.sh/uv/) for dependencies management
- Use [ruff](https://docs.astral.sh/ruff/) as a linter and code formatter

## Usage
Run the main script
```sh
python -m app.main
```