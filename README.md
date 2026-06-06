# Human Physical Activity Action Detection using Machine Learning
> **Classifying Human Motions from Wearable Accelerometer & Gyroscope Telemetry**

This repository contains a Jupyter Notebook implementation for predicting and classifying human actions based on tri-axial accelerometer and gyroscope sensor readings. The model evaluates several classification algorithms to accurately detect distinct actions (e.g., standing still, sitting, walking, etc.).

---

## Project Overview

Detecting human activities using wearable sensor telemetry is a key component of modern health-tracking devices, smart assistants, and sports science applications. This project builds a supervised classification pipeline that maps continuous sensor signals to specific physical postures and movements.

### Activities Detected:
- `None` (Idle/Resting)
- `Standing still`
- `Sitting and relaxing`
- `Lying down`
- `Walking`
- `Climbing stairs`
- `Running`
- `Cycling`
- ... and other standard physical exercises.

---

## Notebook Workflow

The implementation is structured inside [`Human_Action_Detection.ipynb`](Human_Action_Detection.ipynb) as follows:

1. **Libraries & Data Loading**:
   - Imports libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, and `statsmodels`.
   - Loads the sensor dataset.
2. **Exploratory Data Analysis (EDA) & Data Balancing**:
   - Inspects the distribution of different actions.
   - **Downsampling**: Balances the class imbalance by downsampling the heavily-represented `Activity = 0` (None) class to 40,000 records.
3. **Data Encoding & Scaling**:
   - Encodes categorical targets and subject IDs using `LabelEncoder`.
   - Visualizes feature ranges via box plots.
   - Evaluates preprocessing using `StandardScaler` and `RobustScaler` to normalize feature magnitudes.
4. **Model Training & Evaluation**:
   - Splits data into 75% train and 25% test.
   - Implements a custom performance evaluator (`resultsSummarizer`) which outputs accuracy, macro precision, recall, F1-score, and plots a confusion matrix heatmap.
   - Trains and compares multiple classifiers:
     - **Logistic Regression**: Baseline classification score of `~55.45%` test accuracy (due to non-linear relationships in raw signals).
     - **K-Nearest Neighbors (KNN)**: Achieved exceptionally high performance on normalized features:
       - **$K=1$**: Accuracy: **`95.39%`** | Precision: `95.23%` | Recall: `95.50%` | F1-Score: `95.29%`
       - **$K=2$**: Accuracy: **`94.88%`** | Precision: `94.68%` | Recall: `95.09%` | F1-Score: `94.75%`
     - **Decision Tree Classifier**: Built with depth constraints to control complexity.

---

## Installation & Usage

### 1. Install Dependencies
Make sure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```
Open [`Human_Action_Detection.ipynb`](Human_Action_Detection.ipynb) and run all cells.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
