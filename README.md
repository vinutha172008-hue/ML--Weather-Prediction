# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset
To develop a model using the Random Forest Algorithm to predict temperature, PM2.5 level, and energy consumption based on environmental sensor data like humidity, wind speed, and pressure.

Dataset: The dataset contains environmental parameters such as: Humidity Wind Speed Pressure Temperature PM2.5 Energy Format: CSV file (weather_data.csv) Type: Numerical data


## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the programe
2. Import the required libraries such as pandas and sklearn.
3. Load the environmental sensor dataset from the CSV file.
4. Select Humidity, WindSpeed and Pressure as input features.
5. Select Temperature, PM2.5 and Energy as output variables.
6. Split the dataset into training and testing data.
7. Initialize the Random Forest Regressor model.
8. Train the model using the training dataset
9. Predict the output values using the test dataset
10. Display the predicted values.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: VINUTHA V
RegisterNumber: 212225230306 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

data = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
data.columns = data.columns.str.strip()
data['time'] = pd.to_datetime(data['time'])
data = data.sort_values(by='time').reset_index(drop=True)
columns_fill = [
    'tem', 'pm2_5', 'tsr', 'hum',
    'pressure', 'wind_speed',
    'illumination', 'co2'
]

for column in columns_fill:
    if column in data.columns:
        data[column] = data[column].interpolate(
            method='linear',
            limit=10
        )
data['hour'] = data['time'].dt.hour
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
targets = ['tem', 'pm2_5', 'tsr']
for value in targets:
    data[f'{value}_lag1'] = data[value].shift(1)
processed_data = data.dropna(
    subset=['tem_lag2', 'pm2_5_lag2', 'tsr_lag2', 'hum', 'pressure']
).reset_index(drop=True)
processed_data.to_csv(
    "processed_weather_dataset.csv",
    index=False
)

feature_list = [
    'hum',
    'pressure',
    'wind_speed',
    'illumination',
    'co2',
    'hour_sin',
    'hour_cos',
    'tem_lag1',
    'pm2_5_lag1',
    'tsr_lag1'
]

print("\n------ Feature Engineering Summary ------")
print("Total Rows :", len(data))
print("Processed Rows :", len(processed_data))
print("Selected Features :", feature_list)

split_value = int(len(processed_data) * 0.8)

train_data = processed_data.iloc[:split_value]
test_data = processed_data.iloc[split_value:]

X_train = train_data[feature_list]
X_test = test_data[feature_list]
models = {}
results = {}

target_info = {
    'tem': ('Temperature', '°C', 'red'),
    'pm2_5': ('PM2.5 Pollution', 'µg/m³', 'green'),
    'tsr': ('Energy (Solar Radiation)', 'W/m²', 'orange')
}

for target in targets:

    y_train = train_data[target]
    y_test = test_data[target]

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    predictions = rf_model.predict(X_test)

    models[target] = rf_model

    results[target] = {
        'r2_score': r2_score(y_test, predictions),
        'mae_score': mean_absolute_error(y_test, predictions),
        'predicted': predictions,
        'actual': y_test.values
    }

fig, axes = plt.subplots(3, 2, figsize=(16, 18))

for i, target in enumerate(targets):

    label, unit, color = target_info[target]
    output = results[target]
    axes[i, 0].plot(
        output['actual'][-150:],
        label='Actual',
        color='black',
        linewidth=2,
        alpha=0.5
    )

    axes[i, 0].plot(
        output['predicted'][-150:],
        label='Predicted',
        color=color,
        linestyle='--',
        linewidth=2
    )

    axes[i, 0].set_title(
        f"{label}: Actual vs Predicted\n"
        f"R2 Score: {output['r2_score']:.3f} | "
        f"MAE: {output['mae_score']:.2f}"
    )

    axes[i, 0].set_ylabel(unit)
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)

    importance_values = pd.Series(
        models[target].feature_importances_,
        index=feature_list
    ).sort_values()

    importance_values.plot(
        kind='barh',
        ax=axes[i, 1],
        color=color,
        alpha=0.7
    )

    axes[i, 1].set_title(
        f"Important Features - {label}"
    )

plt.tight_layout()
plt.show()

latest_row = processed_data.iloc[-1]

future_input = pd.DataFrame([{
    'hum': latest_row['hum'],
    'pressure': latest_row['pressure'],
    'wind_speed': latest_row['wind_speed'],
    'illumination': latest_row['illumination'],
    'co2': latest_row['co2'],
    'hour_sin': latest_row['hour_sin'],
    'hour_cos': latest_row['hour_cos'],
    'tem_lag1': latest_row['tem'],
    'pm2_5_lag1': latest_row['pm2_5'],
    'tsr_lag1': latest_row['tsr']
}])

print("\n------ Future Predictions ------")

for target in targets:

    predicted_value = models[target].predict(
        future_input
    )[0]

    print(
        f"Predicted {target_info[target][0]} : "
        f"{predicted_value:.2f} "
        f"{target_info[target][1]}"
    )
```

## Output:
![alt text](<Screenshot 2026-05-13 130917.png>)


## Result:
Thus the program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm verified successfully
