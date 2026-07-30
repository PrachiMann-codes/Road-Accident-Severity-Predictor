# 🚦 Road Traffic Accident Severity Predictor

## 📌 Overview
This project predicts the severity of road traffic accidents (Slight, Serious, Fatal) using machine learning techniques.  
It leverages a dataset of **12,316 records with 32 features** including driver demographics, vehicle details, road conditions, and accident circumstances.

---

## 📂 Dataset
- **Source**: RTA_Dataset.csv  
- **Shape**: 12,316 rows × 32 columns  
- **Target Variable**: `Accident_severity`  
  - Slight Injury: 84.56%  
  - Serious Injury: 14.15%  
  - Fatal Injury: 1.28%

---

## ⚙️ Steps Implemented

### 1. Data Loading
- Mounted Google Drive in Colab
- Loaded CSV using `pandas`

### 2. Data Understanding
- Checked shape, column names, datatypes
- Identified missing values (e.g., `Defect_of_vehicle` ~36%, `Service_year_of_vehicle` ~32%)

### 3. Preprocessing
- **Missing values**: Imputed categorical with mode, numeric with median
- **Time conversion**: Extracted `Hour` and `Time_of_day` buckets
- **Encoding**: Label encoding for categorical features, scaling for numeric
- **Target mapping**:  
  - Slight Injury → 0  
  - Serious Injury → 1  
  - Fatal Injury → 2

### 4. Feature Engineering
- Dropped outcome-leak columns (`Casualty_class`, `Casualty_severity`, etc.)
- Created correlation heatmap
- Addressed **class imbalance** using **SMOTE**

### 5. Exploratory Data Analysis (EDA)
- Accident severity distribution
- Weather conditions vs severity
- Lighting conditions analysis
- Road surface & alignment analysis
- Vehicles involved vs severity
- Day of week vs severity

---

## 📊 Key Insights
- Majority of accidents result in **Slight Injuries**.
- **Weather** and **Lighting conditions** strongly influence severity.
- Fatal accidents are rare but critical to predict.
- SMOTE balanced the dataset to improve model training.

---

## 🛠️ Tech Stack
- **Languages**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn, Flask
- **Environment**: Google Colab

---
