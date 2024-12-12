# Feature Selection for Dementia Prediction

This folder contains scripts and tools for **feature selection** applied to the task of dementia prediction using speech data. The methods implemented here were explored as part of the research on Alzheimer's disease (AD) detection using machine learning (ML) algorithms, focusing on acoustic feature extraction from speech. 

### Overview
Feature selection is a crucial step in machine learning, particularly in high-dimensional datasets like those derived from speech data. By identifying and retaining the most relevant features, the models' performance can be improved while reducing complexity and computational requirements. This folder provides implementations of various feature selection techniques, which were evaluated as part of the study titled *"PREDIÇÃO DA DEMÊNCIA POR MEIO DA FALA USANDO ALGORITMOS DE SELEÇÃO DE ATRIBUTOS E APRENDIZADO DE MÁQUINA"*.

This study was later published in the **5th Congresso de Engenharias e Ciências Aplicadas das Três Fronteiras**. You can access the proceedings of the conference [here](https://drive.google.com/file/d/1pueWZ_Hv3mV-kSPWV2Wf0Mb5Er68fGL1/view) for those interested in reading the full paper.

### Contents
The folder contains the following files:

- **`01_data_preprocessing.py`**: Handles data loading, preprocessing, and feature extraction. 
- **`02_feature_selection.py`**: Implements the feature selection methods.
- **`03_model_training.py`**: Contains code for training different machine learning models.
- **`04_main.py`**: The entry point for running the entire pipeline, combining preprocessing, feature selection, and model training.
- **`05_utils.py`**: Includes helper functions for data manipulation, evaluation metrics, and visualization.
- **`06_notebook.ipynb`**: Jupyter notebook for interactive exploration, experimentation, and visualization.
- **`07_adress_20_db_*`**: Dataset files, including training and testing sets for different feature sets (e.g., **eGeMAPS**, **ComParE**, **emobase**).

### Feature Selection Methods
- **Univariate Feature Selection (UFS)**: This method selects features based on statistical tests. It evaluates the relationship between each feature and the target variable, selecting the most relevant features based on the scores.
- **Recursive Feature Elimination (RFE)**: RFE is a wrapper method that recursively removes features and builds models on the remaining features. The process is repeated until the specified number of features is reached.
- **SelectFromModel (SFM)**: SFM selects features based on the importance weights assigned by a model, such as Random Forests or Support Vector Machines.
- **Sequential Feature Selection (SFS)**: SFS is a greedy algorithm that sequentially selects the best features. It can be forward or backward, and it works by adding (or removing) features based on model performance.

### Data Used
The dataset used in this project is from the **ADReSS 2020 Interspeech Challenge**, which contains speech recordings from 78 control group participants and 78 individuals diagnosed with Alzheimer's disease. The speech features were extracted using the openSMILE toolkit, and include several feature sets such as **eGeMAPS**, **ComParE**, and **emobase**.

### Dependencies
The following libraries are required:
- `scikit-learn` for machine learning algorithms and feature selection methods.
- `numpy` for numerical operations.
- `pandas` for data handling.
- `openSMILE` for feature extraction (if you are working with raw speech data).
- `matplotlib` for visualizing results (optional).

### How to Use
1. Clone or download this repository.
2. Install the required dependencies using `pip`:
   ```bash
   pip install scikit-learn numpy pandas openSMILE matplotlib
   ```
3. To use feature selection on your own dataset, modify the example usage script (`example_usage.py`) to load your speech data and apply the feature selection techniques. You can adjust the number of features or the method used to select features by modifying the function parameters.
   
4. After selecting the features, use them in your machine learning models (e.g., Support Vector Machine, Random Forest) for training and evaluation.

### Results
The research in the paper showed that feature selection has a significant impact on model performance. For example, the Sequential Feature Selection (SFS) method consistently provided the best results in terms of accuracy across different feature sets.

The following results were observed with various classification algorithms:
- **SVM** and **Random Forest** performed best with the Sequential Subset Selection method, particularly with the **emobase** feature set.
- **LDA** was sensitive to the number of features selected, with better performance observed at 30% of the features compared to using the full set.
- Reducing the number of features generally improved model accuracy by eliminating irrelevant or redundant features.

### Acknowledgements
We acknowledge the **Dementia Bank** for providing the dataset, and the **UNILA** (Universidade Federal da Integração Latino Americana) for supporting this research.