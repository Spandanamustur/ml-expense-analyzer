import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("data.csv")
df = df.drop(columns=["ID", "ZIP Code"], errors='ignore')

# Target creation (use income + ccavg)
def classify_spending(income, ccavg):
    if income > 80 and ccavg > 2:
        return "High"
    elif income > 40:
        return "Medium"
    else:
        return "Low"

df["Spending_Class"] = df.apply(
    lambda row: classify_spending(row["Income"], row["CCAvg"]), axis=1
)

# ✅ KEEP INCOME in features
X = df.drop("Spending_Class", axis=1)
y = df["Spending_Class"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = KNeighborsClassifier(n_neighbors=11)
model.fit(X_scaled, y)

# UI
st.title("💰 Smart Expense Behaviour Analyzer")
st.markdown("### Enter User Details")

# Inputs
age = st.number_input("Age (18 - 100)", min_value=18, max_value=100, value=25)
experience = st.number_input("Work Experience (years)", min_value=0, max_value=50, value=2)

income = st.number_input("Monthly Income (in thousands ₹)", min_value=0, value=50,
                         help="Example: 50 means ₹50,000")
st.info(f"Entered Income: {income}")

family = st.number_input("Family Members (1 - 10)", min_value=1, max_value=10, value=2)

ccavg = st.number_input("Average Credit Card Spend per month", value=1.0,
                        help="Example: 1.5 means moderate usage")

education = st.selectbox(
    "Education Level",
    [1, 2, 3],
    format_func=lambda x: {
        1: "Undergraduate",
        2: "Graduate",
        3: "Advanced/Professional"
    }[x]
)

mortgage = st.number_input("Mortgage Amount", value=0)

loan = st.selectbox("Personal Loan (0 = No, 1 = Yes)", [0, 1])
sec_acc = st.selectbox("Securities Account (0 = No, 1 = Yes)", [0, 1])
cd_acc = st.selectbox("CD Account (0 = No, 1 = Yes)", [0, 1])
online = st.selectbox("Uses Online Banking (0 = No, 1 = Yes)", [0, 1])
credit = st.selectbox("Has Credit Card (0 = No, 1 = Yes)", [0, 1])

# Income message (just info, not prediction)
if income > 80:
    st.write("Income is High")
elif income > 40:
    st.write("Income is Medium")
else:
    st.write("Income is Low")

# Prediction
if st.button("Predict Spending Behavior"):
    user_data = [[age, experience, income, family, ccavg, education,
                  mortgage, loan, sec_acc, cd_acc, online, credit]]

    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)

    st.success(f"Predicted Spending Class: {prediction[0]}")
