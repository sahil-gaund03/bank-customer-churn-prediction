<div align="center">

# 🏦 Bank Customer Churn Prediction

### End-to-End ML System · Behavioral Analytics · Real-Time Streamlit Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-189fdd?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge)](https://bank-customer-churn-prediction0310.streamlit.app/)

<br/>

> **Predict which bank customers are about to leave — before they do.**  
> A production-ready machine learning project built on behavioral feature engineering, a tuned Random Forest pipeline, and an interactive Streamlit analytics dashboard with live churn prediction.

<br/>

[![Live Demo](https://img.shields.io/badge/▶_Open_Live_Dashboard-bank--customer--churn--prediction0310.streamlit.app-FF4B4B?style=flat-square&logo=streamlit)](https://bank-customer-churn-prediction0310.streamlit.app/)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [ML Architecture](#-ml-architecture--workflow)
- [Business KPIs](#-business-kpis)
- [Model Performance](#-model-performance)
- [Key Insights](#-key-insights-from-the-data)
- [Folder Structure](#-folder-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Screenshots](#-screenshots)
- [Deployment](#-deployment)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🧠 Overview

Customer churn costs banks billions annually. Traditional approaches rely on demographics alone — this project goes further.

**Bank Customer Churn Prediction** uses behavioral analytics to surface *why* customers disengage, not just *who* is at risk. It combines a production-ready Scikit-learn pipeline with an interactive Streamlit dashboard, enabling real-time churn probability scoring and segment-level business insights.

**What makes this different:**

- 🔬 Six hand-crafted behavioral features beyond raw demographics
- 🏗️ End-to-end Scikit-learn pipeline (preprocessing → model → inference)
- 📊 Five actionable business KPIs tracked live in the dashboard
- 🎯 High-risk customer detector (disengaged + high-balance segment)
- 🔮 Live churn prediction from custom input in the browser

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Data** | Pandas, NumPy | Data wrangling and feature engineering |
| **Visualization** | Matplotlib, Seaborn, Plotly | EDA and dashboard charts |
| **ML** | Scikit-learn, XGBoost | Pipeline, preprocessing, classification |
| **Explainability** | SHAP | Feature importance and model transparency |
| **Serialization** | Joblib | Model persistence (`.pkl`) |
| **Dashboard** | Streamlit | Interactive analytics UI |
| **Deployment** | Streamlit Cloud | Live hosting |

---

## ✨ Features

<details>
<summary><strong>🔹 Behavioral Feature Engineering</strong></summary>

Six new features derived from raw bank data to capture engagement and risk signals:

| Feature | Formula | Signal |
|---|---|---|
| `EngagementScore` | `IsActiveMember × NumOfProducts` | Overall engagement level |
| `BalanceSalaryRatio` | `Balance / (EstimatedSalary + 1)` | Financial health indicator |
| `TenureAgeRatio` | `Tenure / (Age + 1)` | Loyalty relative to life stage |
| `ProductUsageLevel` | Binned from `NumOfProducts` | Low / Medium / High segmentation |
| `IsHighValueCustomer` | `Balance > 75th percentile` | Top-tier customer flag |
| `IsDisengagedHighValue` | High-value AND inactive | Highest-priority churn risk |

</details>

<details>
<summary><strong>🔹 Machine Learning Pipeline</strong></summary>

- End-to-end **Scikit-learn Pipeline** (preprocessing + model in one object)
- **StandardScaler** for numeric features
- **OneHotEncoder** for `Geography` and `Gender`
- **Random Forest Classifier** with **GridSearchCV** hyperparameter tuning
- Pipeline serialized to `churn_model.pkl` — ready to load and serve instantly

</details>

<details>
<summary><strong>🔹 Interactive Streamlit Dashboard</strong></summary>

Four sections, all filterable via a dynamic sidebar:

| Section | Description |
|---|---|
| **1️⃣ Churn Overview** | Churn rate, distribution charts, KPI metrics |
| **2️⃣ Engagement Analysis** | Activity vs. churn bar charts, engagement score histogram |
| **3️⃣ Product Utilization** | Churn by product count, usage segment breakdown |
| **4️⃣ High-Risk Detector** | Live table of disengaged high-balance customers |
| **🔮 Live Prediction** | Input customer data → real-time churn probability |

**Sidebar filters:** Age range · Balance range · Number of products · Active member status

</details>

---

## 🏗 ML Architecture & Workflow

```
Raw Data (European_Bank.csv)
        │
        ▼
┌───────────────────────┐
│   Feature Engineering │  ← 6 behavioral features derived
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│   Scikit-learn        │
│   Pipeline            │
│  ┌─────────────────┐  │
│  │ StandardScaler  │  │  ← Numeric columns
│  │ OneHotEncoder   │  │  ← Geography, Gender
│  └─────────────────┘  │
│  ┌─────────────────┐  │
│  │  Random Forest  │  │  ← GridSearchCV tuned
│  └─────────────────┘  │
└───────────────────────┘
        │
        ▼
  churn_model.pkl
        │
        ▼
┌───────────────────────┐
│  Streamlit Dashboard  │  ← KPIs · Charts · Live prediction
└───────────────────────┘
```

---

## 📈 Business KPIs

Five real-time KPIs tracked in the dashboard, recalculated dynamically as filters change:

| KPI | Abbreviation | Definition |
|---|---|---|
| Engagement Retention Ratio | **ERR** | Retention rate among active members |
| Product Depth Index | **PDI** | Average number of products held per customer |
| High-Balance Disengagement Rate | **HBDR** | Inactivity rate among top-quartile balance customers |
| Credit Card Stickiness | **CCS** | Churn rate specifically among credit card holders |
| Relationship Strength Index | **RSI** | Mean engagement score across the filtered segment |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | ~85% |
| **ROC-AUC** | > 0.80 |
| **Precision** | High (low false positive rate) |
| **Recall** | Strong churn detection sensitivity |
| **CV Strategy** | Stratified K-Fold |
| **Tuning** | GridSearchCV on key Random Forest hyperparameters |

> Model trained on 10,000 customer records from a European bank dataset. Evaluation uses stratified splits to account for class imbalance (~20% churn rate).

---

## 💡 Key Insights from the Data

| Finding | Detail |
|---|---|
| 🔴 **Inactive customers churn 2–3× more** | `IsActiveMember` is among the strongest predictors in the model |
| 🟢 **2 products = peak retention** | Customers with exactly two products show the lowest churn rate |
| ⚠️ **High-balance + inactive = highest risk** | This segment is flagged in real time by the High-Risk Detector |
| 💳 **Credit card ownership slightly improves retention** | Minor but consistent signal across the dataset |

---

## 📁 Folder Structure

```
bank-customer-churn-prediction/
│
├── European_Bank.csv          # Source dataset — 10,000 customer records
├── churn_analysis.ipynb       # Full EDA, feature engineering, model training
├── churn_model.pkl            # Serialized Scikit-learn pipeline (Random Forest)
├── streamlit_app.py           # Streamlit dashboard — main entry point
├── requirements.txt           # Python dependencies
└── README.md                  # You are here
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/sahil-gaund03/bank-customer-churn-prediction.git
cd bank-customer-churn-prediction

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501` by default.

---

## 🚀 Usage

### Running the Dashboard

```bash
streamlit run streamlit_app.py
```

Use the **sidebar** to filter by age range, balance range, number of products, and member activity status. All KPIs and charts update in real time.

### Making a Prediction

1. Scroll to the **🔮 Predict Customer Churn** section
2. Enter customer details: Age, Balance, Credit Score, Salary, Products, Activity
3. Click **Predict**
4. The model returns a churn probability and flags the customer as high or low risk

### Running the Notebook

```bash
jupyter notebook churn_analysis.ipynb
```

The notebook covers the full workflow: EDA → feature engineering → pipeline construction → evaluation.

---

## 🖼 Screenshots

> _Add screenshots to a `/screenshots` folder and update the paths below._

| Dashboard View | Description |
|---|---|
| ![KPI Overview](https://via.placeholder.com/800x400?text=KPI+Metrics+Banner) | Live KPI cards: ERR, PDI, HBDR, CCS, RSI |
| ![Churn Overview](https://via.placeholder.com/800x400?text=Churn+Distribution+Chart) | Churn distribution and churn rate |
| ![Engagement Analysis](https://via.placeholder.com/800x400?text=Engagement+Analysis) | Activity vs. churn, engagement score histogram |
| ![High-Risk Detector](https://via.placeholder.com/800x400?text=High-Risk+Customer+Table) | Real-time high-risk customer table |
| ![Prediction Widget](https://via.placeholder.com/800x400?text=Live+Churn+Prediction) | Live churn probability output |

---

## ☁️ Deployment

### Streamlit Cloud (Current)

The app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `streamlit_app.py`
5. Click **Deploy**

🔗 **Live:** [bank-customer-churn-prediction0310.streamlit.app](https://bank-customer-churn-prediction0310.streamlit.app/)

---

<details>
<summary><strong>Docker (Optional)</strong></summary>

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t bank-churn-app .
docker run -p 8501:8501 bank-churn-app
```

</details>

---

## 🔮 Future Improvements

| Area | Planned Enhancement |
|---|---|
| **Explainability** | SHAP value plots per prediction (global + local importance) |
| **Model Comparison** | XGBoost and LightGBM benchmarking with leaderboard |
| **API Layer** | FastAPI endpoint for real-time scoring from external systems |
| **Cloud** | AWS SageMaker or Render deployment with CI/CD pipeline |
| **Monitoring** | Data drift detection with Evidently AI |
| **Containerization** | Docker + docker-compose for reproducible environments |

---

## 🤝 Contributing

Contributions are welcome. To contribute:

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Commit your changes with a clear message
git commit -m "feat: add SHAP explanation panel to dashboard"

# 4. Push and open a Pull Request
git push origin feature/your-feature-name
```

**Before submitting a PR:**
- Ensure the app runs without errors (`streamlit run streamlit_app.py`)
- Keep feature engineering consistent with training (columns must match `churn_model.pkl`)
- Add a brief description of your change in the PR

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

<div align="center">

**Sahil Gaund**

*Data Science · Machine Learning · Analytics Engineering*

[![GitHub](https://img.shields.io/badge/GitHub-sahil--gaund03-181717?style=for-the-badge&logo=github)](https://github.com/sahil-gaund03)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-sahilgaund03-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/sahilgaund03)
[![Portfolio](https://img.shields.io/badge/Portfolio-sahilgaund0310.netlify.app-00C7B7?style=for-the-badge&logo=netlify)](https://sahilgaund0310.netlify.app/)

</div>

---

## ⭐ Support

If this project helped you or you found it useful:

- **Star the repo** — it helps others discover the project
- **Fork it** — build your own version or extend the analysis
- **Share it** — with anyone learning ML, data science, or analytics

```
⭐ Star this repo if it was useful to you!
```

<div align="center">

---

*Built with Python · Scikit-learn · Streamlit · SHAP*

</div>
