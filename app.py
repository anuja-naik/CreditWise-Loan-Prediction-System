import streamlit as st
import pickle
import numpy as np

log_model=pickle.load(open("logistic_model.pkl","rb"))
knn_model=pickle.load(open("knn_model.pkl","rb"))
nb_model=pickle.load(open("nb_model.pkl","rb"))
scaler=pickle.load(open("scaler.pkl","rb"))

st.title("CreditWise Loan Prediction System")

Applicant_Income=st.number_input("Applicant Income")

Coapplicant_Income=st.number_input("Coapplicant Income")

Age=st.number_input("Age")

Dependents=st.number_input("Dependents")

Existing_Loans=st.number_input("Existing Loans")

Savings=st.number_input("Savings")

Collateral_Value=st.number_input("Collateral Value")

Loan_Amount=st.number_input("Loan Amount")

Loan_Term=st.number_input("Loan Term")

Credit_Score=st.number_input("Credit Score")

DTI_Ratio=st.number_input("DTI Ratio")

Credit_Score_sq=Credit_Score**2

DTI_Ratio_sq=DTI_Ratio**2

input_data=np.array([[

Applicant_Income,
Coapplicant_Income,
Age,
Dependents,
Existing_Loans,
Savings,
Collateral_Value,
Loan_Amount,
Loan_Term,

0,

1,
0,
0,

1,

0,
0,
1,
0,

0,
1,

1,

0,
1,
0,
0,

DTI_Ratio_sq,
Credit_Score_sq

]])

if st.button("Predict"):

    scaled=scaler.transform(input_data)

    st.write(
        "Logistic:",
        "Approved"
        if log_model.predict(scaled)[0]==1
        else "Rejected"
    )

    st.write(
        "KNN:",
        "Approved"
        if knn_model.predict(scaled)[0]==1
        else "Rejected"
    )

    st.write(
        "Naive Bayes:",
        "Approved"
        if nb_model.predict(scaled)[0]==1
        else "Rejected"
    )