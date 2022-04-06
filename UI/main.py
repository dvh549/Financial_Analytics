# to run: type 'streamlit run UI/main.py' in command line

# change according to relative path to this file

import sklearn
import pickle
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import scorecardpy as sc

# rf_model = pickle.load(open(file_location + 'predictors/xgb_pkl.pkl', 'rb'))
# xgb_woe_model = pickle.load(open(file_location + 'predictors/xgb_woe.pkl', 'rb'))
# xgb_model = pickle.load(open(file_location + 'UI/predictors/xgb.pkl', 'rb'))
svml_model = pickle.load(open('UI/predictors/svm_linear_kernel_oversampled.pkl', 'rb'))

# possible inputs sorted list according to encoded values
gender = ["M", "F"]
family = ['Civil Marriage', 'Married', 'Single / Not Married', 'Separated', 'Widow']
jobs = ['Accountants', 'Cleaning Staff', 'Cooking Staff', 'Core Staff', 'Drivers', 'HR Staff', 'High Skill Tech Staff', 'IT Staff', 'Laborers', 'Low-Skill Laborers', 'Managers', 'Medicine Staff', 'Private Service Staff', 'Realty Agents', 'Sales Staff', 'Secretaries', 'Security Staff', 'Unemployed', 'Waiters / Barmen Staff']
education = ['Lower Secondary', 'Secondary / Special Secondary', 'Incomplete Higher', 'Higher Education', 'Academic Degree']
housing = ['Co-op apartment', 'House / apartment', 'Municipal apartment	', 'Office apartment', 'Rented apartment', 'With parents']
income = ['Commercial Associate', 'Pensioner', 'State Servant', 'Student', 'Working']


header_pic = Image.open('UI/images/header.png')
st.image(header_pic, use_column_width=True)
st.title("Calculate Credit Default Risk")

st.subheader("Answer the following questions about the customer.")

# Main block asking questions about user

# Gender
gender_list = sorted(gender)
gender_st = st.selectbox("Gender", options=gender_list)

# age
age = st.number_input("What is the customer's age?", value=0, step=1)

# num years with us
cust_mths = st.number_input("How many months has the customer been with us?", value=0, step=1)

# assets
st.text("Select the assets that the customer own:")
col1, col2= st.columns(2)

with col1:
    car = st.checkbox('Car')
    email = st.checkbox('Email')
    phone = st.checkbox('Phone')
with col2:
    realty = st.checkbox('Realty')
    work_phone = st.checkbox('Work Phone')

# Family
family_list = sorted(family)
family_st = st.selectbox("What is the customer's family type type?", options=family_list)
fam_count = st.number_input("How many people are in the customer's family?", value=0, step=1)
children = st.number_input("How many children does the customer have?", value=0, step=1)

# job
job_list = sorted(jobs)
job_st = st.selectbox("What is the customer's occupation type?", options=job_list)
emp_years = st.number_input("How many years has the customer worked for?", value=0, step=1)
income_list = sorted(income)
income_st = st.selectbox("What is the customer's income type?", options=income_list)
income = st.number_input("What is your income?", value=0, step=100)

# education
education_list = sorted(education)
education_st = st.selectbox("What is the customer's highest education level?", options=education_list)

# housing
housing_list = sorted(housing)
housing_st = st.selectbox("What is your housing type?", options=housing_list)

# Processing Functions
def category_encode_func(category_list, input):
    i = category_list.index(str(input))
    return i
        
def yes_no_encode(input):
    if input:
        return 1
    else: 
        return 0


# Credit Risk Summary
if st.button("Click to generate credit risk summary"):
    # encode all the inputs
    gender = category_encode_func(gender, gender_st)
    car = yes_no_encode(car)
    realty = yes_no_encode(realty)
    children = int(children)
    income_amt = int(income)
    income_type = category_encode_func(income_list, income_st)
    education_type = category_encode_func(education, education_st)
    family_status = category_encode_func(family, family_st)
    housing_type = category_encode_func(housing, housing_st)
    work_phone = yes_no_encode(work_phone)
    phone = yes_no_encode(phone)
    email = yes_no_encode(email)
    job_type = category_encode_func(jobs, job_st)
    fam_count = int(fam_count)
    cust_mths = int(cust_mths)
    emp_years = int(emp_years)
    age = int(age)

    # Build the dict of inputs
    print(job_st.lower().capitalize())
    sc_data = {'FLAG_OWN_REALTY': [realty],
        'EMP_YEARS': [emp_years],
        'AGE': [age],
        'AMT_INCOME_TOTAL': [income_amt],
        'NAME_FAMILY_STATUS': [family_st.lower().capitalize()],
        'OCCUPATION_TYPE': [job_st.lower().capitalize()],
        'CUST_FOR_MONTHS': [cust_mths]
    }

    data = {'CODE_GENDER': [gender], 
            'FLAG_OWN_CAR': [car],
            'FLAG_OWN_REALTY': [realty],
            'CNT_CHILDREN': [children],
            'AMT_INCOME_TOTAL': [income_amt],
            'NAME_INCOME_TYPE': [income_type],
            'NAME_EDUCATION_TYPE': [education_type],
            'NAME_FAMILY_STATUS': [family_status],
            'NAME_HOUSING_TYPE': [housing_type],
            'FLAG_WORK_PHONE': [work_phone],
            'FLAG_PHONE': [phone],
            'FLAG_EMAIL': [email],
            'OCCUPATION_TYPE': [job_type],
            'CNT_FAM_MEMBERS': [fam_count],
            'CUST_FOR_MONTHS': [cust_mths],
            'EMP_YEARS': [emp_years],
            'AGE': [age]
            }

    info_df = pd.DataFrame.from_dict(data)
    sc_df = pd.DataFrame.from_dict(sc_data)
    # print(sc_df)
    
    # prediction = rf_model.predict(info_df)[0]
    # prediction = xgb_model.predict(info_df)[0]
    # print("prediction:", prediction)

    


    

    # if prediction == 0:
    #     st.write("customer is unlikely to default")
        
    # if prediction == 1:
    #     st.write("Customer is likely to default")

    path = 'UI/scorecard/'

    f = open( path + 'woe_bins.txt', 'r')
    cat = f.readlines()
    bins_dict = {}
    for c in cat:
        c = c.rstrip('\n')
        df_cat = pd.read_csv( path + c + '.csv')
        bins_dict[c] = df_cat
    f.close()

    lr_model = pickle.load(open('UI/scorecard/woe_lr.pkl', 'rb'))
    f = open( path + 'x_train_columns.txt', 'r')
    x_train_columns = []
    cols = f.readlines()
    for c in cols:
        x_train_columns.append(c.rstrip('\n'))
    f.close()
    # info_filtered = info_df[bins_dict.keys()]
    card = sc.scorecard(bins_dict, lr_model, x_train_columns, basepoints_eq0 = 0, points0 = 600, odds0 = 1/50, pdo = 20)
    data_woe = sc.woebin_ply(sc_df, bins_dict)
    # card = sc.scorecard(bins_dict, lr_model, x_train_columns)
    test_score = sc.scorecard_ply(sc_df, card, only_total_score=False)
    
    # print(test_score['score'][0])
    # print(test_score.loc[0, ['score']])
    final_score = test_score['score'][0]

    def determine_score(score):
        if score < 580:
            return "a very bad"
        elif score < 670:
            return "a fair"
        elif score < 740:
            return "a good"
        elif score < 800:
            return "a very good"
        else:
            return "an exceptional"
    
    rating = determine_score(final_score)

    prediction = svml_model.predict(data_woe)[0]
    print(prediction)

    default_msg = ''

    if prediction == 0:
        default_msg = "The customer is **unlikely** to default. "
        
    if prediction == 1:
        default_msg = "The customer is **likely** to default. "
    
    st.write( default_msg + "He has " + rating + " credit score of "+ "**" + str(int(final_score))   + "**" +". The table below shows the breakdown of the score: ")

    # test_score['BASE SCORE'] = 600
    test_score.insert(0, "BASE_SCORE_points", [600])
    # convert individual feature points to int
    for column in test_score.columns:
        test_score[column] = int(test_score[column])
    test_score["FINAL_SCORE_points"] = test_score["score"]
    test_score.drop(["score"], axis=1, inplace=True)
    test_score.insert(0, "SCORE_BREAKDOWN", ["CUSTOMER_SCORE_BREAKDOWN"])
    test_score.set_index("SCORE_BREAKDOWN", inplace=True)
    print(test_score)
    st.table(test_score.loc["CUSTOMER_SCORE_BREAKDOWN", :])
    
    #  st.image("https://static.streamlit.io/examples/dice.jpg")

