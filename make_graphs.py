import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('./GBM Attendance - Meeting data.csv')

data['Meeting'] = pd.to_datetime(data['Meeting'])

plt.figure(figsize=(12, 6))
plt.plot(data['Meeting'], data['Attendance'], color='blue', marker='', linestyle='-')
plt.xlabel("Date")
plt.ylabel("Attendance")
plt.title("Attendance Over the Years")

plt.xticks(rotation=45)

plt.grid(True)

plt.show()