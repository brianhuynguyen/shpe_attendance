from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

df['Weight'] = df['Year'].apply(lambda x: 6 if x == 2024 else (5 if x == 2023 else (4 if x == 2022 else 0.5)))

X = df[['Month', 'Day', 'Year', 'Week of the Semester', 'Season Encoded', 'Discord Messages', 'First GBM', 'Last Meeting Attendance', 'Room Capacity','Post Convention']]
y = df['Attendance']  # Target

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, df['Weight'], test_size=0.2, random_state=42)

# Model 1: Ridge Regression
ridge_model = Ridge(random_state=42)
ridge_model.fit(X_train, y_train, sample_weight=weights_train)
y_pred_ridge = ridge_model.predict(X_test)
print("Ridge Regression:")
print("MAE:", mean_absolute_error(y_test, y_pred_ridge))
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("R² Score:", str("{:.2%}".format(r2_score(y_test, y_pred_ridge))))

# # Model 2: Random Forest Regressor
# rf_model = RandomForestRegressor(random_state=42)
# rf_model.fit(X_train, y_train)
# y_pred_rf = rf_model.predict(X_test)
# print("\nRandom Forest Regressor:")
# print("MAE:", mean_absolute_error(y_test, y_pred_rf))
# print("MSE:", mean_squared_error(y_test, y_pred_rf))
# print("R² Score:", r2_score(y_test, y_pred_rf))

# # Model 3: Lasso Regression
# lasso_model = Lasso(random_state=42)
# lasso_model.fit(X_train, y_train)
# y_pred_lasso = lasso_model.predict(X_test)
# print("\nLasso Regression:")
# print("MAE:", mean_absolute_error(y_test, y_pred_lasso))
# print("MSE:", mean_squared_error(y_test, y_pred_lasso))
# print("R² Score:", r2_score(y_test, y_pred_lasso))

# # Model 4: Linear Regression
# linear_model = LinearRegression()
# linear_model.fit(X_train, y_train)
# y_pred_linear = linear_model.predict(X_test)
# print("\nLinear Regression:")
# print("MAE:", mean_absolute_error(y_test, y_pred_linear))
# print("MSE:", mean_squared_error(y_test, y_pred_linear))
# print("R² Score:", r2_score(y_test, y_pred_linear))

# Prediction
today = datetime.now()
days_until_monday = (7 - today.weekday()) % 7

next_gbm = today + timedelta(days=days_until_monday)

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
    'Week of the Semester': 5, 
    'Season Encoded': season_encoded,       
    'Discord Messages': 29,    # Fill this out
    'First GBM': first_gbm,            
    'Last Meeting Attendance': data['Attendance'][len(data['Attendance'])-1],
    'Room Capacity': 80,
    'Post Convention': 0
}

upcoming_gbm_df = pd.DataFrame([upcoming_gbm])

predicted_attendance = ridge_model.predict(upcoming_gbm_df)

print(f'Predicted Attendance for Upcoming GBM on {next_gbm.date()}: {round(predicted_attendance[0])}')

