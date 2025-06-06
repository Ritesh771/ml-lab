import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)


clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(15, 10))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.title("Decision Tree for Breast Cancer Classification")
plt.show()

new_sample = np.array([
    [14.0, 20.5, 90.0, 600.0, 0.1, 0.2, 0.3, 0.15, 0.25, 0.05,
     0.5, 1.0, 3.0, 40.0, 0.005, 0.02, 0.02, 0.01, 0.02, 0.003,
     16.0, 25.0, 100.0, 800.0, 0.15, 0.3, 0.4, 0.2, 0.3, 0.07]
])

prediction = clf.predict(new_sample)
print(f"New Sample Classification: {'Malignant' if prediction[0] == 0 else 'Benign'}")
