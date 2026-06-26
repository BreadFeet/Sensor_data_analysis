# Sensor Data Analysis

## Overview

This project analyzes wearable sensor data to **classify exercise types and estimate repetition counts** using machine learning techniques. The workflow covers data preprocessing, visualization, feature engineering, model training, and evaluation.

---

## Main Workflow

The notebooks below should be followed in this order.

### 1. `make_dataset.ipynb`

Loads the raw sensor files, merges accelerometer and gyroscope measurements, and creates the initial dataset.

**[Columns]**

* `set`: Identifier assigned to each raw data file. The numbering is arbitrary and does not reflect the chronological order of data collection. It is used only to separate data points belonging to the same exercise recording.

### 2. `visualize.ipynb`

Visualizes sensor signals, feature distributions, and intermediate results for exploratory data analysis.

### 3. `remove_outliers.ipynb`

Detects and removes noisy observations using multiple outlier detection methods.

### 4. `build_features.ipynb`

Creates statistical, temporal, and frequency domain features for model training.

### 5. `train_model.ipynb`

Trains and evaluates multiple machine learning models for exercise classification.

### 6. `count_repetitions.ipynb`

Estimates exercise repetitions from processed sensor signals.

---

## Supporting Modules

These Python modules provide reusable functions used throughout the notebooks.

| File | Description |
| --- | --- |
| `DataTransformation.py` | Functions for preprocessing and feature transformation. |
| `TemporalAbstraction.py` | Generates time domain features using rolling windows. |
| `FrequencyAbstraction.py` | Extracts frequency domain features using the Discrete Fourier Transform (DFT). |
| `LearningAlgorithms.py` | Implements machine learning models and evaluation methods. |

---

## Environment Setting

Install the required packages using the `environment.yml` file.

```bash
conda env create -f environment.yml
conda activate <environment_name>
```

---

## Things to Think About

* Which feature engineering steps are safe to apply to the entire dataset, and which should be performed only on the training set?
* Should low pass filtering and rolling window operations be applied to the entire dataset or separately for each `set`?
* How should the data be split into training and test sets? Random shuffling or participant based hold out?
