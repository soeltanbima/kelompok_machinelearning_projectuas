import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import os

st.title("Prediksi Gaji Karyawan Berdasarkan Pengalaman")
for dirname, _, filenames in os.walk('/Project_ml'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

salary = pd.read_csv("Salary.csv")
salary_df = pd.DataFrame(salary)
salary_df.head()

X = salary_df.iloc[:, :-1].values
y = salary_df.iloc[:, -1].values
X

plt.scatter(X, y)
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)
predictions

print(y_test)

plt.scatter(X_train, y_train, color="red")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.plot(X_train, regressor.predict(X_train))

mean_absolute_error(y_test, predictions)