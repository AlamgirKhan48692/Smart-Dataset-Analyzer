# 📊 Smart Dataset Analyzer

An end-to-end Data Analysis and Machine Learning web application built with Flask, Pandas, Scikit-Learn, Matplotlib, and Seaborn.

This platform allows users to upload datasets, analyze data, handle missing values, generate visualizations, perform statistical analysis, and automatically train machine learning models without writing code.

---

## 🎥 Project Demo

[![Project Demo](screenshots/upload.png)](https://youtu.be/ulhbAIGRqPM)

---

Live Demo

🚀 Live Application: https://smart-dataset-analyzer-app.onrender.com

---

## 🚀 Features

### 📁 Dataset Upload
- Upload CSV datasets
- Automatic dataset loading
- Quick dataset preview

### 📋 Dataset Information
- First rows preview
- Dataset shape
- Dataset description
- Data types information
- Missing values analysis

### 🧹 Missing Value Treatment
- Detect missing values automatically
- Numerical columns:
  - Mean Imputation
  - Median Imputation
- Categorical columns:
  - Mode Imputation
- Download cleaned dataset

### 📈 Statistical Analysis
- Mean
- Median
- Mode
- Summary Statistics

### 📊 Data Visualization
#### Single Column Analysis
- Histogram
- Pie Chart
- Bar Chart
- Box Plot

#### Relationship Analysis
- Scatter Plot
- Line Chart

#### Dataset Analysis
- Correlation Heatmap
- Correlation Matrix
- Missing Value Analysis

### 🤖 Machine Learning
Automatically detects problem type and trains machine learning models.

Supported Models:

#### Regression
- Linear Regression

#### Classification
- Logistic Regression
- Decision Tree
- Random Forest

### 📉 Prediction Results
- Model Evaluation
- Best Model Selection
- Prediction Comparison
- Actual vs Predicted Values

---

# 🖼️ Application Workflow

```text
Upload Dataset
        ↓
Dataset Information
        ↓
Missing Value Detection
        ↓
Missing Value Treatment
        ↓
Statistical Analysis
        ↓
Data Visualization
        ↓
Machine Learning Training
        ↓
Prediction Results
```

---

# 📸 Screenshots

## Upload Dataset

![Upload Dataset](screenshots/upload.png)

---

## Dataset Overview

![Dataset Overview](screenshots/dataset_info_a.png)
![Dataset Overview](screenshots/dataset_info_b.png)

---

## Missing Value Detection

![Missing Value Detection](screenshots/missing_values.png)

---

## Missing Value Treatment

![Missing Value Treatment](screenshots/treatment.png)

---

## Cleaned Dataset

![Cleaned Dataset](screenshots/cleaned_dataset.png)

---

## Types of Columns

![Types of Columns](screenshots/columns_type.png)

---

## Statistical Analysis

![Statistics](screenshots/statistics.png)

---

## Visualization Dashboard

![Visualization Dashboard](screenshots/visualization_dashboard.png)

---

## Correlation Heatmap

![Heatmap](screenshots/heatmap.png)

---

## Histogram Analysis

![Histogram](screenshots/histogram.png)

---

## Horizontal Bar Chart

![Horizontal Bar Chart](screenshots/horizontal_bar_chart.png)

---

## Scatter Plot Analysis

![Scatter Plot](screenshots/scatter_plot.png)

---

## Machine Learning Results

![Machine Learning](screenshots/ml_result.png)

---

## Prediction Results

![Prediction Results](screenshots/predictions.png)

---

# 🛠️ Technologies Used

### Backend
- Python
- Flask

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-Learn

### Data Visualization
- Matplotlib
- Seaborn

### Frontend
- HTML
- CSS
- JavaScript

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Smart-Dataset-Analyzer.git
```

Move into project directory:

```bash
cd Smart-Dataset-Analyzer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

# 📂 Project Structure

```text
Smart-Dataset-Analyzer/
│
├── static/
│   ├── css/
│   ├── images/
│
├── templates/
│
├── uploads/
│
├── screenshots/
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🎯 Future Improvements

- User Authentication
- Excel File Support (.xlsx)
- PDF Report Generation
- XGBoost Integration
- KNN Classification
- SVM Classification
- Model Comparison Dashboard
- Interactive Charts (Plotly)
- Dark Mode UI
- Cloud Deployment

---

# 🌟 Why This Project?

This project simplifies the complete data science workflow for beginners, students, and analysts by combining:

- Data Cleaning
- Data Analysis
- Data Visualization
- Machine Learning

into a single web application.

---

# 👨‍💻 Author

**Alamgir Khan**

GitHub: https://github.com/AlamgirKhan48692

---

## ⭐ If you find this project useful, consider giving it a star.
