from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

f = open('GBM Attendance - Meeting data.csv', 'r')

data = {}

for i, line in enumerate(f):
    if i == 0:
        line = line.strip().split(',')
        for cat in line:
            data[cat]= []
    else:
        line = line.split(',')
        data['Meeting'].append(line[0])
        data['Attendance'].append(line[1])
        data['Month'].append(line[2])
        data['Day'].append(line[3])
        data['Year'].append(line[4])
        data['Week of the Semester'].append(line[5])
        data['Season'].append(line[6])
        data['Season Encoded'].append(line[7])
        data['Discord Messages'].append(line[8])
        data['First GBM'].append(line[9])
        data['Last Meeting Attendance'].append(line[10])
        data['Room Capacity'].append(line[11])
        data['Post Convention'].append(line[12].strip())

df = pd.DataFrame(data)

df['Meeting'] = pd.to_datetime(df['Meeting'])
df['Attendance'] = pd.to_numeric(df['Attendance'])
df['Month'] = pd.to_numeric(df['Month'])
df['Day'] = pd.to_numeric(df['Day'])
df['Year'] = pd.to_numeric(df['Year'])
df['Week of the Semester'] = pd.to_numeric(df['Week of the Semester'])
df['Season Encoded'] = pd.to_numeric(df['Season Encoded'])
df['Discord Messages'] = pd.to_numeric(df['Discord Messages'])
df['First GBM'] = pd.to_numeric(df['First GBM'])
df['Last Meeting Attendance'] = pd.to_numeric(df['Last Meeting Attendance'])
df['Room Capacity'] = pd.to_numeric(df['Room Capacity'])
df['Post Convention'] = pd.to_numeric(df['Post Convention'])

df['Weight'] = df['Year'].apply(lambda x: 3 if x == 2024 else (3 if x == 2023 else (2 if x == 2022 else 1)))

X = df[['Month', 
        'Day',
        'Year',
        'Week of the Semester',
        'Season Encoded',
        'First GBM',
        'Last Meeting Attendance',
        ]]

y = df['Attendance']
weights = df['Weight']

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, df['Weight'], test_size=0.2, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores, mae_scores, mse_scores = [], [], []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    weights_train = weights.iloc[train_index]
    
    ridge_model = Ridge()
    ridge_model.fit(X_train, y_train, sample_weight=weights_train)
    
    y_pred = ridge_model.predict(X_test)
    r2_scores.append(r2_score(y_test, y_pred))
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    mse_scores.append(mean_squared_error(y_test, y_pred))

# Display cross-validation results
print(f"Mean R² score: {np.mean(r2_scores):.2%}")
print(f"Mean MAE: {np.mean(mae_scores)}")
print(f"Mean MSE: {np.mean(mse_scores)}")


# Prediction
prev_gbm = df['Meeting'].iloc[-1]
next_gbm = prev_gbm + timedelta(days=7)
if next_gbm.month>7:
    season_encoded = 0
else:
    season_encoded = 1
if next_gbm.month == 9 and next_gbm.day <= 16:
    first_gbm = 1
else:
    first_gbm = 0

upcoming_gbm = {
    'Month': next_gbm.month,               
    'Day': next_gbm.day,                 
    'Year': next_gbm.year,              
    'Week of the Semester': 8, 
    'Season Encoded': season_encoded,
    'First GBM': first_gbm,            
    'Last Meeting Attendance': data['Attendance'][len(data['Attendance'])-1],
}

upcoming_gbm_df = pd.DataFrame([upcoming_gbm])
predicted_attendance = ridge_model.predict(upcoming_gbm_df)
print(f'Predicted Attendance for Upcoming GBM on {next_gbm.date()}: {round(predicted_attendance[0])}')

