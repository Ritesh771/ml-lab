import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
x_values = np.random.rand(100).reshape(-1, 1)
labels = np.array(["Class1" if x <= 0.5 else "Class2" for x in x_values[:50]])
k_values = [1, 2, 3, 4, 5, 20, 30]
for k in k_values:
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(x_values[:50], labels)
	predicted_labels = knn.predict(x_values[50:])	
	print(f"Results for k={k}:")
	print(predicted_labels)
	print("\n")	
	plt.scatter(x_values[:50], np.zeros(50), c=["blue" if x <= 0.5 else "red" for x in x_values[:50]], label='Labeled Data')
	plt.scatter(x_values[50:], np.zeros(50), c='green', marker='x', label='Unlabeled Data')
	plt.xlabel("x values")
	plt.title("KNN Classification of Randomly Generated Data")
	plt.legend()
plt.show()