# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:38:15 2021

@author: ATBI-0533
"""

from sklearn import datasets
import numpy as np
import pandas as pd
df = pd.read_csv('diabetes.csv')
x=df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',
      'DiabetesPedigreeFunction','Age']]
y=df.Outcome
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc=sc.fit(x)
X = sc.transform(x)
print(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
# l1 regularization gives better results
lr = LogisticRegression(C=10, random_state=0)
lr.fit(X_train, y_train)
from sklearn import metrics
# generate evaluation metrics
print("Train - Accuracy :", metrics.accuracy_score(y_train, lr.predict(X_train)))
print("Train - Confusion matrix :",metrics.confusion_matrix(y_train,lr.predict(X_train)))
print("Train - classification report :", metrics.classification_report(y_train, lr.predict(X_train)))
print("Test - Accuracy :", metrics.accuracy_score(y_test, lr.predict(X_test)))
print("Test - Confusion matrix :",metrics.confusion_matrix(y_test,lr.predict(X_test)))
print("Test - classification report :", metrics.classification_report(y_test, lr.predict(X_test)))

import streamlit as st

st.title('Diabetes Prediction APP')
st.sidebar.header('Enter Your Details to predict diabetes')

def user_report():
     Pregnancies = st.sidebar.slider('Pregnancies', 0,5, 15 )
     Glucose = st.sidebar.slider('Glucose', 0,100, 210 )
     BloodPressure = st.sidebar.slider('BloodPressure', 0,60, 125 )
     SkinThickness = st.sidebar.slider('SkinThickness', 0,50, 100 )
     Insulin = st.sidebar.slider('Insulin', 0,425, 850 )
     BMI = st.sidebar.slider('BMI', 0,33, 67 )
     DiabetesPedigreeFunction = float(st.sidebar.slider('DiabetesPedigreeFunction', 0.0,0.1, 3.0 ))
     Age = st.sidebar.slider('Age', 20,35, 81 )
     user_report_data = {
      'Pregnancies':Pregnancies,
      'Glucose':Glucose,
      'BloodPressure':BloodPressure,
      'SkinThickness':SkinThickness,
      'Insulin':Insulin,
      'BMI':BMI,
      'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
      'Age':Age      
      }

     report_data = pd.DataFrame(user_report_data, index=[0])
     return report_data

user_data = user_report()

st.header('Applicants Data')

st.write(user_data)

Diabetes = lr.predict(user_data)
st.write(Diabetes)
st.subheader('Your diabetes Status ')
if (Diabetes==0):
    st.subheader('YES')
else:
    st.subheader('NO')
    












