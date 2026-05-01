import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib 
import matplotlib.pyplot as plt 
import seaborn as sns 

st.set_page_config(page_title="Bank Churn Dashboard", layout="wide")

# load data + model 

#------------------------------------------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("European_Bank.csv")
    
    # feture engineering (must match training )
    df['EngagementScore'] = df['IsActiveMember'] * df['NumOfProducts']
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['TenureAgeRatio'] = df['Tenure'] / (df['Age'] + 1)

    df['ProductUsageLevel'] = pd.cut(
        df['NumOfProducts'],
        bins=[0,1,2,4],
        labels=['Low','Medium','High']
    )
    
    threshold = df['Balance'].quantile(0.75)
    df['IsHighValueCustomer'] = (df['Balance'] > threshold).astype(int)

    
    df['IsDisengagedHighValue'] = (
        (df['IsHighValueCustomer'] == 1) &
        (df['IsActiveMember'] == 0)
    ).astype(int)

    return df

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

df = load_data()
model = load_model()

#------------------------------------------------------------------------------------------------

# SIDEBAR FILTERS

st.sidebar.header("🔍 Filter Customers")

age_range = st.sidebar.slider("Age", int(df.Age.min()), int(df.Age.max()), (20,50))
balance_range = st.sidebar.slider("Balance", int(df.Balance.min()), int(df.Balance.max()), (0,100000))
products_filter = st.sidebar.multiselect("NumOfProducts", df.NumOfProducts.unique(), default=df.NumOfProducts.unique())
active_filter = st.sidebar.multiselect("IsActiveMember", [0,1], default=[0,1])

#------------------------------------------------------------------------------------------------

# Apply filters
filtered_df = df[
    (df['Age'].between(age_range[0], age_range[1])) &
    (df['Balance'].between(balance_range[0], balance_range[1])) &
    (df['NumOfProducts'].isin(products_filter)) &
    (df['IsActiveMember'].isin(active_filter))
]

#------------------------------------------------------------------------------------------------

# KPIs
def calculate_kpis(data):

    ERR = 1 - data[data['IsActiveMember']==1]['Exited'].mean()
    PDI = data['NumOfProducts'].mean()
    HBDR = data[data['IsHighValueCustomer']==1]['IsActiveMember'].value_counts(normalize=True).get(0,0)
    CCS = data[data['HasCrCard']==1]['Exited'].mean()
    RSI = data['EngagementScore'].mean()

    return ERR, PDI, HBDR, CCS, RSI

ERR, PDI, HBDR, CCS, RSI = calculate_kpis(filtered_df)

#------------------------------------------------------------------------------------------------

# TITLE
st.title("🏦 Banking Customer Churn Dashboard")

#------------------------------------------------------------------------------------------------

# KPI DISPLAY
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Engagement Retention", f"{ERR:.2f}")
col2.metric("Product Depth", f"{PDI:.2f}")
col3.metric("High Balance Risk", f"{HBDR:.2f}")
col4.metric("Credit Card Churn", f"{CCS:.2f}")
col5.metric("Relationship Strength", f"{RSI:.2f}")

#------------------------------------------------------------------------------------------------

# SECTION 1: CHURN OVERVIEW
st.header("1️⃣ Churn Overview")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.countplot(x='Exited', data=filtered_df, ax=ax)
    ax.set_title("Churn Distribution")
    st.pyplot(fig)

with col2:
    churn_rate = filtered_df['Exited'].mean()
    st.write(f"### Churn Rate: {churn_rate:.2%}")
    
#------------------------------------------------------------------------------------------------
    
# SECTION 2: ENGAGEMENT ANALYSIS
st.header("2️⃣ Engagement Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.barplot(x='IsActiveMember', y='Exited', data=filtered_df, ax=ax)
    ax.set_title("Churn vs Activity")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['EngagementScore'], kde=True, ax=ax)
    ax.set_title("Engagement Score Distribution")
    st.pyplot(fig)
    
#------------------------------------------------------------------------------------------------
    
# SECTION 3: PRODUCT UTILIZATION
st.header("3️⃣ Product Utilization")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.barplot(x='NumOfProducts', y='Exited', data=filtered_df, ax=ax)
    ax.set_title("Churn vs Number of Products")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.countplot(x='ProductUsageLevel', data=filtered_df, ax=ax)
    ax.set_title("Product Usage Segments")
    st.pyplot(fig)

# SECTION 4: HIGH-RISK CUSTOMERS
st.header("4️⃣ High-Risk Customer Detector")

high_risk_df = filtered_df[
    filtered_df['IsDisengagedHighValue'] == 1
]

st.write(f"High-Risk Customers Found: {len(high_risk_df)}")

if high_risk_df.empty:
    st.warning("No high-risk customers match current filters")
else:
    st.dataframe(high_risk_df.head(20), use_container_width=True)
    
#------------------------------------------------------------------------------------------------

# MODEL PREDICTION
st.subheader("🔮 Predict Customer Churn")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 80, 30)
    balance = st.number_input("Balance", 0, 200000, 50000)

with col2:
    products = st.selectbox("NumOfProducts", [1,2,3,4])
    active = st.selectbox("IsActiveMember", [0,1])

with col3:
    credit_score = st.number_input("CreditScore", 300, 900, 600)
    salary = st.number_input("EstimatedSalary", 10000, 200000, 50000)

#------------------------------------------------------------------------------------------------
    
# Create input
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Geography':['France'],
    'Gender':['Male'],
    'Age':[age],
    'Tenure':[5],
    'Balance':[balance],
    'NumOfProducts':[products],
    'HasCrCard':[1],
    'IsActiveMember':[active],
    'EstimatedSalary':[salary],

    # ✅ MATCH TRAINING FEATURES EXACTLY
    'EngagementScore':[active * products],
    'BalanceSalaryRatio':[balance / (salary + 1)],

    # ⚠️ FIX NAME (was TenureAgeRatio)
    'TenureRatio':[5 / (age + 1)],

    'ProductUsageLevel':['Medium'],
    'IsHighValueCustomer':[1 if balance > df['Balance'].quantile(0.75) else 0],
    'IsDisengagedHighValue':[1 if (balance > df['Balance'].quantile(0.75) and active==0) else 0],

    # ⚠️ ADD MISSING COLUMN
    'Year':[2024]   # or whatever default used in training
})

if st.button("Predict"):
    prob = model.predict_proba(input_data)[0][1]

    st.write(f"### Churn Probability: {prob:.2f}")

    if prob > 0.5:
        st.error("⚠️ High Risk Customer")
    else:
        st.success("✅ Low Risk Customer")