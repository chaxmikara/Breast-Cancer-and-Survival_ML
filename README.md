# Breast Cancer and Survival Analysis - Machine Learning Project

This project focuses on analyzing breast cancer patient data to predict survival outcomes using machine learning techniques. The analysis includes comprehensive data preprocessing, exploratory data analysis, and the development of both classification and regression models.

## 📋 Project Overview

This machine learning project analyzes breast cancer patient data to:
- **Classification Task**: Predict mortality status (Alive/Dead)
- **Regression Task**: Predict survival months for deceased patients
- Perform comprehensive data exploration and visualization
- Apply various preprocessing techniques including outlier detection and feature encoding

## 📁 Project Structure

```
Breast Cancer and Survival_ML/
├── README.md
├── Datasets/
│   ├── 5DATA002W.2 Coursework Dataset(25012025v6.0).csv  # Original dataset
│   ├── classification_dataset_cleaned.csv               # Processed data for classification
│   └── regression_dataset_cleaned.csv                   # Processed data for regression
├── 5DATA002W_2_Final_Python_Notebook_1.ipynb           # Data preprocessing & EDA
├── 5DATA002W_2_Final_Python_Notebook_2_Classification_Modelling.ipynb  # Classification models
├── 5DATA002W_2_Final_Python_Notebook_3_Classification_Modelling.ipynb  # Additional classification analysis
└── Breast_Cancer_and_Survival.ipynb                    # Main analysis notebook
```

## 📊 Dataset Description

The dataset contains **4,025 breast cancer patient records** with the following features:

### Patient Demographics
- **Age**: Patient age
- **Sex**: Gender (Male/Female)
- **Month_of_Birth**: Birth month

### Tumor Characteristics
- **T_Stage**: Primary tumor stage (T1, T2, T3, T4)
- **N_Stage**: Regional lymph node involvement (N0, N1, N2, N3)
- **6th_Stage**: Overall cancer stage (IIA, IIIA, IIIC, etc.)
- **A_Stage**: Anatomical stage classification
- **Tumor_Size**: Size of the primary tumor
- **Grade**: Tumor grade (1-3)
- **Differentiated**: Degree of cell differentiation

### Biomarkers
- **Estrogen_Status**: Estrogen receptor status (Positive/Negative)
- **Progesterone_Status**: Progesterone receptor status (Positive/Negative)

### Treatment & Outcomes
- **Regional_Node_Examined**: Number of lymph nodes examined
- **Reginol_Node_Positive**: Number of positive lymph nodes
- **Survival_Months**: Survival time in months
- **Mortality_Status**: Patient outcome (Alive/Dead)

## 🛠️ Data Preprocessing

The preprocessing pipeline includes:

### 1. Data Cleaning
- **Missing Value Handling**: Imputation of missing values in categorical variables
- **Outlier Detection**: IQR method for identifying and correcting outliers
- **Data Type Corrections**: Converting age and tumor size to appropriate data types
- **Value Standardization**: Normalizing categorical values (e.g., mortality status)

### 2. Feature Engineering
- **Categorical Encoding**: Label encoding for categorical variables
- **Feature Scaling**: StandardScaler for numerical features
- **Feature Selection**: Removal of unnecessary columns (Patient_ID, Month_of_Birth, Occupation)

### 3. Data Quality Issues Addressed
- Corrected erroneous age values (502 → 52, 180 → 80, -50 → 50)
- Fixed negative tumor sizes by taking absolute values
- Standardized gender encoding (1 → Male)
- Normalized mortality status formatting

## 📈 Exploratory Data Analysis

Key visualizations include:
- **Age Distribution**: Histogram showing patient age demographics
- **Mortality Status**: Bar chart of survival outcomes
- **Tumor Size Distribution**: Box plot revealing size patterns
- **Correlation Matrix**: Heatmap of numerical variable relationships
- **Categorical Analysis**: Unique value distributions for staging variables

## 🤖 Machine Learning Tasks

### Classification Task
- **Objective**: Predict mortality status (binary classification)
- **Target Variable**: Mortality_Status (0 = Alive, 1 = Dead)
- **Dataset**: Complete cleaned dataset

### Regression Task
- **Objective**: Predict survival months for deceased patients
- **Target Variable**: Survival_Months
- **Dataset**: Subset of patients with Mortality_Status = 1 (Dead)

## 🔧 Dependencies

```python
# Core Data Science Libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import preprocessing
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required Python packages (see Dependencies section)

### Installation
1. Clone this repository
2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn
   ```
3. Open the Jupyter notebooks in order:
   - Start with `5DATA002W_2_Final_Python_Notebook_1.ipynb` for data preprocessing
   - Continue with classification modeling notebooks

### Running the Analysis
1. **Data Preprocessing**: Run the first notebook to clean and prepare the data
2. **Classification Modeling**: Execute the classification notebooks to build predictive models
3. **Results Analysis**: Review model performance and insights

## 📋 Usage

### Data Loading
```python
import pandas as pd
df = pd.read_csv('Datasets/5DATA002W.2 Coursework Dataset(25012025v6.0).csv')
```

### Accessing Cleaned Data
```python
# For classification tasks
classification_data = pd.read_csv('Datasets/classification_dataset_cleaned.csv')

# For regression tasks
regression_data = pd.read_csv('Datasets/regression_dataset_cleaned.csv')
```

## 🎯 Key Findings

- Dataset contains diverse breast cancer cases with varying stages and characteristics
- Successful identification and correction of data quality issues
- Comprehensive preprocessing pipeline prepared data for machine learning
- Both classification and regression datasets created for different analytical objectives

## 📝 Notes

- Original dataset contains 4,025 patient records
- Data spans various tumor stages and patient demographics
- Preprocessing ensures data quality and model readiness
- Multiple notebooks provide modular analysis approach

## 🤝 Contributing

This project is part of coursework 5DATA002W.2. For contributions or questions, please refer to the course guidelines.

## 📄 License

This project is for educational purposes as part of university coursework.

---

*Last updated: July 2025*
