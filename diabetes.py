import streamlit as st
import pandas as pd
import sklearn
import pickle
st.title(":rainbow[Diabetes Prediction on the basis of given input]")
st.markdown(""":rainbow[ Accuracy For This Model  = 98% \nPrecision Score is 98% and 100%        ]""")
# gender	age	hypertension	heart_disease	smoking_history	bmi	HbA1c_level	blood_glucose_level
gender=st.selectbox(label=":rainbow[Gender| Male=1,Female=0,other=2]",options=[0,1,2])
age=st.number_input(label=":rainbow[Your age]",step=1)
hypertension=st.selectbox(":rainbow[Hypertension|No=0,Yes=1]",options=[0,1])
# 0    35816
# 4    35095
# 3     9352
# 1     9286
# 5     6447
# 2     4004
# No Info        35816
# never          35095
# former          9352
# current         9286
# not current     6447
# ever
smoking_history=st.selectbox(":rainbow[Smoking_history|current=1,ever_smoking=2,former=3,never=4,no info=0,Not current=5]",options=[0,1,2,3,4,5])
heart_disease=st.selectbox(":rainbow[heart_disease|Yes=1,No=0]",options=[0,1])
bmi=st.number_input(":rainbow[Bmi]")
HbA1c_level=st.number_input(":rainbow[HbA1c_level]")
blood_glucose_level=st.number_input(":rainbow[blood_glucose_level]")
df=pd.DataFrame({"gender":[gender],"age":	[age],"hypertension":	[hypertension],"hear_disease":	[heart_disease],"smoking_history":	[smoking_history]	,"bmi":[bmi],"hba1c_level":	[HbA1c_level],"blood_glucose_level":	[blood_glucose_level]})
st.write(df)
# Assuming you have a saved model file named 'your_model.pkl'
model_file_path = 'mo.pkl'

# Load the model from the file
with open(model_file_path, 'rb') as file:
    loaded_model = pickle.load(file, fix_imports=False)
values=loaded_model.predict(df.values)
if st.button("click me to check Prediction"):
    if values==1:
        st.write(":red[Prediction yes diabetes]")
    else:
        st.write(":green[Prediction No diabetes]")
