import numpy as np 
import pandas as pd
import time


def read_data(train = r"D:\Acads\Courses\DA2401\Endsem\MNIST_train.csv", val = r"D:\Acads\Courses\DA2401\Endsem\MNIST_validation.csv"):
    traindata = pd.read_csv(train)
    valdata = pd.read_csv(val)
    Xtrain = traindata.iloc[:, 1:-1].values
    ytrain = traindata.iloc[:, 0].values
    Xval = valdata.iloc[:, 1:-1].values
    yval = valdata.iloc[:, 0].values
    return Xtrain/255.0, ytrain.astype(int), Xval/255.0, yval.astype(int)


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis = 0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar = False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors_sorted = eigenvectors[:, idx]
        self.components = eigenvectors_sorted[:, :self.n_components]
        self.explained_variance_ratio = eigenvalues[idx][:self.n_components]/np.sum(eigenvalues)

    def transform(self, X):
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA not fitted")
        X_centered = X - self.mean
        X_projected = X_centered @ self.components
        return X_projected
    

class KNNClassifier:
    def __init__(self, n_neighbors = 5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def _predict_single(self, x):
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._euclidean_distance(x, x_train)
            distances.append((dist, self.y_train[i]))

        distances.sort(key = lambda x: x[0])
        k_neighbors = distances[:self.n_neighbors]
        k_neighbor_labels = [label for _, label in k_neighbors]
        
        most_common = np.bincount(k_neighbor_labels).argmax()
        return most_common
    
    def predict(self, X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

def f1_score(y, y_pred):
    n_classes = len(np.unique(y))
    f1_scores = []
    for c in range(n_classes):
        tp = np.sum((y == c) & (y_pred == c))
        fp = np.sum((y != c) & (y_pred == c))
        fn = np.sum((y==c) & (y_pred != c))
        precision = tp/(tp + fp) if (tp + fp) > 0 else 0
        recall = tp/(tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision*recall)/(precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    return np.mean(f1_scores)


X_train, y_train, X_val, y_val = read_data()

#Preprocessing data with PCA
start_time = time.perf_counter()
pca = PCA(n_components = 100)
pca.fit(X_train)
x_train_pca = pca.transform(X_train)
x_val_pca = pca.transform(X_val)
end_time = time.perf_counter()
print(f"Time taken for PCA: {end_time - start_time:.2f} seconds")

#Making Predictions with KNN
start_time = time.perf_counter()
knn = KNNClassifier(n_neighbors = 6)
knn.fit(x_train_pca, y_train)
y_pred_knn = knn.predict(x_val_pca)
end_time = time.perf_counter()


print(f"Time taken for Final Prediction (FULL DATA): {end_time - start_time:.2f} seconds")
print(f"F1 Score: {f1_score(y_val, y_pred_knn):.4f}")
print(f"Val Accuracy: {np.sum(y_pred_knn == y_val)/len(y_val):.4f}")