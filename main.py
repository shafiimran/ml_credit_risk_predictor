import streamlit as st
from prediction_helper import predict  # Ensure this is correctly linked to your prediction_helper.py

# Set the page configuration and title
st.set_page_config(page_title="Credit Risk Predictor")
st.title("Credit Risk Predictor")

# Create rows of three columns each
    

# Assign inputs to the first row with default values
st.subheader("Applicant Details")
row1 = st.columns(3)
with row1[0]:
    age = st.number_input('Age', min_value=18, step=1, max_value=100, value=28)
with row1[1]:
    income = st.number_input('Income', min_value=0, value=1200000)

with row1[2]:
    residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])

row12 = st.columns(3)
with row12[0]:
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0, max_value=100, step=1, value=30)
with row12[1]:
    num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=4, step=1, value=2)


st.subheader("Loan Information")
row2 = st.columns(3)

with row2[0]:
    loan_amount = st.number_input('Loan Amount', min_value=0, value=2560000)
with row2[1]:
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])
with row2[2]:
    loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'])

row3 = st.columns(1)
# Calculate Loan to Income Ratio and display it
loan_to_income_ratio = loan_amount / income if income > 0 else 0
with row3[0]:
    st.text("Loan to Income Ratio:")
    st.text(f"{loan_to_income_ratio:.2f}")  # Display as a text field

# Assign inputs to the remaining controls


st.subheader("Delinquency Details")
row4 = st.columns(3)
with row4[0]:
    delinquent_months= st.number_input('Delinquent Months', min_value=0, max_value=100, step=1, value=6, help="Number of months the borrower has missed or delayed loan payments")
with row4[1]:
    loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36)
with row4[2]:
    total_dpd = st.number_input('Total DPD', min_value=0, value=30, help="Total number of days the borrower has been late on payments across all loans")

row5 = st.columns(3)
with row5[0]:
    delinquency_ratio = delinquent_months / loan_tenure_months if loan_tenure_months > 0 else 0
    st.text('Delinquency Ratio')
    st.text(f"{delinquency_ratio:.2f}")
    
with row5[1]:
    avg_dpd_per_delinquency = total_dpd / delinquent_months if delinquent_months > 0 else 0
    st.text('Avg DPD')
    st.text(f"{avg_dpd_per_delinquency:.2f}")
    

# Button to calculate risk
if st.button('Calculate Risk'):
    # Call the predict function from the helper module
    # print((age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
    #                                             delinquency_ratio, credit_utilization_ratio, num_open_accounts,
    #                                             residence_type, loan_purpose, loan_type))
    probability, credit_score, rating = predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                                                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                                                residence_type, loan_purpose, loan_type)

    # Display the results
    st.success(f"Default Probability: {probability:.2%}")
    st.success(f"Credit Score: {credit_score}")
    st.success(f"Rating: {rating}")

# Footer
# st.markdown('_Project From Codebasics ML Course_')
