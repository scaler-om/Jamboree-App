import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pickle

# load the dataset
df = pd.read_csv('Jamboree_Admission.csv')

# cleaning the dataset
df = df.drop(columns = 'Serial No.')
df.rename(columns = {'LOR ':'LOR', 'Chance of Admit ': 'Chance of Admit'}, inplace=True)
df['GRE Score'] = df['GRE Score'].astype('int16')
df['TOEFL Score'] = df['TOEFL Score'].astype('int8')
df['University Rating'] = df['University Rating'].astype('int8')
df['SOP'] = df['SOP'].astype('float32')
df['LOR'] = df['LOR'].astype('float32')
df['CGPA'] = df['CGPA'].astype('float32')
df['Research'] = df['Research'].astype('bool')
df['Chance of Admit'] = df['Chance of Admit'].astype('float32')

# prepare data for modeling
df['Research'] = df['Research'].astype('int8')

# train-test split
X = df.drop(columns='Chance of Admit')
y = df[['Chance of Admit']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# data normalization/standardization
columns_to_scale = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']
#Initialize an object of class MinMaxScaler()
min_max_scaler = MinMaxScaler()
# Fit min_max_scaler to training data
min_max_scaler.fit(X_train[columns_to_scale])
# Scale the training and testing data
X_train[columns_to_scale] = min_max_scaler.transform(X_train[columns_to_scale])
X_test[columns_to_scale] = min_max_scaler.transform(X_test[columns_to_scale])

# helper function to check model's performance
def model_performance(y, y_pred, model):
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred,squared = False)
    r2 = r2_score(y, y_pred)
    n = len(y)
    try:
        p = len(model.params)
    except:
        p = len(model.coef_) + len(model.intercept_)
    adj_r2 = 1 - (((1-r2)*(n-1))/(n-p-1))

    print(f'Mean Absolute Error for the model(MAE): {mae:.2f}')
    print(f'Root Mean Squared Error for Model: {rmse:.2f}')
    print(f'R2 Score for Model: {r2:.2f}')
    print(f'Adjusted R2 Score for Model: {adj_r2:.2f}')

# model
regr_lr = LinearRegression()
regr_lr.fit(X_train, y_train)
y_pred_lr = regr_lr.predict(X_test)
model_performance(y_test, y_pred_lr, regr_lr)

# saving the scaler and model
scaler_pkl_file = 'scaler.pkl'
pickle.dump(min_max_scaler, open(scaler_pkl_file, 'wb'))
model_pkl_file = 'model.pkl'
pickle.dump(regr_lr, open(model_pkl_file, 'wb'))
