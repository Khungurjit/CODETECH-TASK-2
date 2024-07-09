import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

data = pd.read_csv('creditcard.csv')

legit = data[data.Class == 0]
fraud = data[data.Class == 1]

legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

model = LogisticRegression()
model.fit(X_train, y_train)

train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

st.title("Credit Card Fraud Detection")

input_df = st.text_input('Enter transaction features')
input_df_lst = input_df.split(',')

submit = st.button("predict")

if submit:
    features = np.array(input_df_lst, dtype=np.float64)
    prediction = model.predict(features.reshape(1,-1))
    
    if prediction[0] == 0:
        st.write("Legit transaction")
    else:
        st.write("Fraud transaction")