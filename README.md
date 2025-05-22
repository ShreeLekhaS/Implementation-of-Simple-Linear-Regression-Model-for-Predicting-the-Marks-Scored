# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students
2. Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored
3. Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis
4. Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b
5. for each data point calculate the difference between the actual and predicted marks
6. Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error
7. Once the model parameters are optimized, use the final equation to predict marks for any new input data

## Program and Output
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Shree Lekha.S
RegisterNumber: 212223110052
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('student_scores.csv')

print(df.head())
```
![image](https://github.com/user-attachments/assets/478d984c-e3b0-4ded-92a5-0c859e797e03)
```
print(df.tail())
```
![image](https://github.com/user-attachments/assets/bc7c378e-f8fd-47cf-9c7b-fa346c3fe102)
```
X = df.iloc[:, :-1].values
print(X)

```
![image](https://github.com/user-attachments/assets/f4dc5290-86c5-4892-a040-29086f9c3295)

```
y = df.iloc[:, -1].values
print(y)
```
![image](https://github.com/user-attachments/assets/d4e59640-f9f5-4d73-ac6c-0a26235f5684)

```

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

```
![image](https://github.com/user-attachments/assets/2f458494-24c7-4961-9a1d-30cf1f679ef1)
```
# Predict test data
y_pred = regressor.predict(X_test)
print(y_pred)
```
![image](https://github.com/user-attachments/assets/8fd6ba17-a24e-470e-881f-26aa1ee23908)
```
print(y_test)

```
![image](https://github.com/user-attachments/assets/e77fce37-356b-417e-89da-48d8057df83a)
```
# Visualize Training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```

![image](https://github.com/user-attachments/assets/e4ff358b-746d-4a1d-b949-d9f98ccb348b)

```
# Visualize Test set
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_pred, color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
![image](https://github.com/user-attachments/assets/f4ced7f3-fab8-4817-a2e5-f1d0c251afb8)

```
# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MSE = {mse:.2f}")
print(f"MAE = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")

```
![image](https://github.com/user-attachments/assets/e13eb456-97f1-4286-9788-729150e987bf)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
