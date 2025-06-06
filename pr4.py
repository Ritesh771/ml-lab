import pandas as pd
import numpy as np

def find_s_algorithm(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    hypothesis = np.array(['?' for _ in range(X.shape[1])])

    for i, row in enumerate(X):
        if y[i] == 'Yes':
            if hypothesis[0] == '?':
                hypothesis = row.copy()
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] != row[j]:
                        hypothesis[j] = '?'

    return hypothesis

data = {
    'Sky': ['Sunny', 'Sunny', 'Overcast', 'Rain'],
    'AirTemp': ['Warm', 'Warm', 'Warm', 'Cold'],
    'Humidity': ['Normal', 'High', 'Normal', 'High'],
    'Wind': ['Strong', 'Strong', 'Weak', 'Weak'],
    'Water': ['Warm', 'Cool', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Change', 'Same', 'Change'],
    'EnjoySport': ['Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

hypothesis = find_s_algorithm(df)
print("The most specific hypothesis is:", hypothesis)
