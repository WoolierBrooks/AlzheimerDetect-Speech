# Feature Selection for Dementia Prediction

This folder contains scripts and tools for **feature selection** applied to the task of dementia prediction using speech data. The methods implemented here were explored as part of the research on Alzheimer's disease (AD) detection using machine learning (ML) algorithms, focusing on acoustic feature extraction from speech.

### Overview
Feature selection is a crucial step in machine learning, particularly when working with high-dimensional datasets like those derived from speech data. By identifying and retaining the most relevant features, the performance of models can be improved, while reducing computational complexity. This folder provides implementations of various feature selection techniques, which were evaluated as part of the study titled *"PREDIÇÃO DA DEMÊNCIA POR MEIO DA FALA USANDO ALGORITMOS DE SELEÇÃO DE ATRIBUTOS E APRENDIZADO DE MÁQUINA"*.

### Contents
The folder contains the following files and scripts:

- **`01_data_preprocessing.py`**: Handles data loading, preprocessing, and feature extraction. 
- **`02_feature_selection.py`**: Implements the feature selection methods.
- **`03_model_training.py`**: Contains code for training different machine learning models.
- **`04_main.py`**: The entry point for running the entire pipeline, combining preprocessing, feature selection, and model training.
- **`05_utils.py`**: Includes helper functions for data manipulation, evaluation metrics, and visualization.
- **`06_notebook.ipynb`**: Jupyter notebook for interactive exploration, experimentation, and visualization.
- **`07_adress_20_db_*`**: Dataset files, including training and testing sets for different feature sets (e.g., **eGeMAPS**, **ComParE**, **emobase**).

### Feature Selection Methods
The following feature selection techniques are implemented:

- **Univariate Feature Selection (UFS)**: Selects features based on statistical tests. It evaluates the relationship between each feature and the target variable, selecting the most relevant ones based on their scores.
- **Recursive Feature Elimination (RFE)**: A wrapper method that recursively removes features and builds models on the remaining features. The process is repeated until the specified number of features is reached.
- **SelectFromModel (SFM)**: Selects features based on the importance weights assigned by a machine learning model, such as Random Forests or Support Vector Machines.
- **Sequential Feature Selection (SFS)**: A greedy algorithm that selects features sequentially. It can be forward (adding features) or backward (removing features), based on model performance.

### Data Used
The dataset used in this project is from the **ADReSS 2020 Interspeech Challenge**, containing speech recordings from 78 control group participants and 78 individuals diagnosed with Alzheimer's disease. The speech features were extracted using the openSMILE toolkit and include multiple feature sets such as **eGeMAPS**, **ComParE**, and **emobase**.

### Dependencies
To run the scripts, the following libraries are required:
- `scikit-learn` for machine learning algorithms and feature selection methods.
- `numpy` for numerical operations.
- `pandas` for data handling.
- `openSMILE` for feature extraction (if working with raw speech data).
- `matplotlib` for visualizations (optional).
  
Install the dependencies using `pip`:
```bash
pip install scikit-learn numpy pandas openSMILE matplotlib
