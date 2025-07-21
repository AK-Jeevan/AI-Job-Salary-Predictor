# This model is to predict the salary of AI jobs based on various features using Multiple Linear Regression.
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('C:/Users/akjee/Documents/ML/ai_job_dataset.csv')

# Remove duplicate rows
data = data.drop_duplicates()

# Remove rows with any missing values
data = data.dropna()

# Display the first few rows of the dataset
print(data.head())

# Display the columns of the dataset
# print(data.columns)

# Display the shape of the dataset i.e number of rows and columns
# print(data.shape)

# Display the statistics of the dataset
print(data.describe())

# To know number of NULL values in each column
# print(data.isnull().sum())
# print(data.info())

scaler = StandardScaler()

# Now define the independent and dependent variables for Multiple Regression
x = data[['experience_level', 'education_required', 'company_size', 'company_location', 'benefits_score']]
x = pd.get_dummies(x, columns=['experience_level', 'education_required', 'company_size', 'company_location'])
x['benefits_score'] = scaler.fit_transform(x[['benefits_score']])

y = data[['salary_usd']] 

# Now divide the dataset into Training set and Testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print(f"X Train shape is :{x_train.shape}")
print(f"X Test shape is :{x_test.shape}")
print(f"Y Train shape is :{y_train.shape}")
print(f"Y Test shape is :{y_test.shape}")

# Now creating the model and fitting it to find patterns
model = LinearRegression()
model.fit(x_train, y_train)

# now to get slope and intercept values for the Independent variables
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Predictions
y_pred = model.predict(x_test)
y_pred = pd.DataFrame(y_pred, columns=['salary_usd'])
print("y_pred shape:",y_pred.shape)

# to check the accuracy of the model we use metrics
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Rsquared :", metrics.r2_score(y_test, y_pred))

# Plotting Actual vs Predicted Salary
plt.figure(figsize=(8,5))
plt.scatter(y_test.values, y_pred.values, color='blue', alpha=0.6)
plt.plot([y_test.values.min(), y_test.values.max()], [y_test.values.min(), y_test.values.max()], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted Salary')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.grid(True)
plt.show()