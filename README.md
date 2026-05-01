# 🏦 Bank Customer Churn Prediction & Analytics Dashboard

A production-ready end-to-end Machine Learning project that predicts customer churn using **behavioral insights, engagement metrics, and product usage patterns**, and delivers actionable insights through an interactive Streamlit dashboard.

---

## 🚀 Project Overview

Customer churn is one of the most critical challenges in banking. Traditional models rely heavily on demographics, but this project focuses on **behavior-driven analytics** to better understand *why customers leave*.

## 🌐 Live Demo

👉https://bank-customer-churn-prediction0310.streamlit.app/

* 📊 Exploratory Data Analysis (EDA)
* ⚙️ Feature Engineering (behavioral metrics)
* 🤖 Machine Learning (Scikit-learn pipeline)
* 📈 KPI-driven business insights
* 🖥️ Interactive Streamlit Dashboard

---

## 🎯 Key Objectives

* Analyze relationship between **customer engagement and churn**
* Evaluate impact of **product usage on retention**
* Identify **high-value but disengaged customers**
* Build a **robust churn prediction model**
* Deliver **actionable business KPIs**
* Enable **real-time prediction via dashboard**

---

## 🧠 Key Features

### 🔹 Behavioral Feature Engineering

* Engagement Score
* Balance-to-Salary Ratio
* Tenure Ratio
* Product Usage Segmentation
* High-Value Customer Identification
* Disengaged High-Value Detection

---

### 🔹 Machine Learning Pipeline

* End-to-end **Scikit-learn Pipeline**
* Automatic preprocessing:

  * Standard Scaling
  * One-Hot Encoding
* Model:

  * Random Forest (with GridSearchCV tuning)
* Evaluation Metrics:

  * Accuracy
  * Precision / Recall
  * F1 Score
  * ROC-AUC

---

### 🔹 Business KPIs

* 📌 Engagement Retention Ratio
* 📌 Product Depth Index
* 📌 High-Balance Disengagement Rate
* 📌 Credit Card Stickiness
* 📌 Relationship Strength Index

---

### 🔹 Interactive Dashboard (Streamlit)

#### Sections:

1. **Churn Overview**
2. **Engagement Analysis**
3. **Product Utilization**
4. **High-Risk Customer Detector**

#### Features:

* 🎛️ Dynamic Sidebar Filters
* 📊 Real-time visualizations
* 🔍 Customer segmentation
* ⚠️ High-risk customer identification
* 🔮 Live churn prediction

---

## 🧱 Project Structure

```
bank_churn_project/
│
├── data/
│   └── European_Bank.csv
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── models/
│   ├── churn_model.pkl
│   └── feature_columns.pkl
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/bank-churn-project.git
cd bank-churn-project
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 📊 Sample Insights

* 🔴 Inactive customers churn **2–3x more**
* 🟢 Customers with **2 products show highest retention**
* ⚠️ High-balance inactive users are the **highest risk segment**
* 💳 Credit card ownership slightly improves retention

---

## 🧪 Model Performance

| Metric    | Value (approx)         |
| --------- | ---------------------- |
| Accuracy  | ~85%                   |
| ROC-AUC   | >0.80                  |
| Precision | High                   |
| Recall    | Strong churn detection |

---

## 📸 Dashboard Preview

*(Add screenshots here for maximum impact)*

---

## 💼 Business Impact

This system enables banks to:

* 🎯 Target high-risk customers proactively
* 📈 Improve retention through engagement strategies
* 💡 Optimize cross-selling opportunities
* 🧠 Make data-driven decisions in real-time

---

## 🔥 Advanced Enhancements (Optional)

* SHAP Explainability for model transparency
* XGBoost / LightGBM comparison
* FastAPI deployment for real-time scoring
* Cloud deployment (AWS / Render)

---

## 🧾 Resume Highlight

> Built an end-to-end ML-powered customer churn prediction system using behavioral analytics and deployed an interactive Streamlit dashboard for real-time risk detection and business insights.

---

## 🤝 Contributing

Feel free to fork, improve, and submit a pull request.

---

## 📬 Contact

For any queries or collaboration opportunities, reach out via GitHub.

---

⭐ If you found this project useful, consider giving it a star!
