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
    return Xtrain, ytrain, Xval, yval

X_train, y_train, X_val, y_val = read_data()

X_train = X_train / 255.0
X_val = X_val / 255.0
y_train = y_train.astype(int)
y_val = y_val.astype(int)

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

class SoftMaxRegression:
    def __init__(self, learning_rate = 0.1, n_iterations = 100, n_classes = 10):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_classes = n_classes
        self.weights = None
        self.bias = None
    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.theta = np.zeros((n_features, self.n_classes))
        self.b = np.zeros((1, self.n_classes))
        y_one_hot = np.zeros((n_samples, self.n_classes))
        y_one_hot[np.arange(n_samples), y] = 1
        for i in range(self.n_iterations):
            Z = X @ self.theta + self.b
            p = self._softmax(Z)
            error = p - y_one_hot
            dtheta = (X.T @ error)/n_samples
            db = np.sum(error, axis = 0, keepdims = True)/n_samples

            self.theta -= self.learning_rate * dtheta
            self.b -= self.learning_rate * db
    def predict(self, X):
        Z = X @ self.theta + self.b
        p = self._softmax(Z)
        return np.argmax(p, axis = 1)

class BaggedClassifier:
    def __init__(self, base_classifier, n_estimators = 10, subsample_ratio = 0.8, subsample_feature_ratio = 0.8):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.subsample_ratio = subsample_ratio
        self.subsample_feature_ratio = subsample_feature_ratio
        self.estimators_and_features = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        subsample_size = int(self.subsample_ratio * n_samples)
        subfeature = int(n_features * self.subsample_feature_ratio)

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, subsample_size, replace = True)
            feature_indices = np.random.choice(n_features, subfeature, replace = False)

            X_sample = X[indices]
            y_sample = y[indices]

            estimator = self.base_classifier()
            estimator.fit(X_sample[:, feature_indices], y_sample)
            self.estimators_and_features.append((estimator, feature_indices))
            print(f"Trained estimator {_ + 1}/{self.n_estimators}")
    def predict(self, X):
        predictions = []
        count = 1
        for estimator, feature_indices in self.estimators_and_features:
            preds = estimator.predict(X[:, feature_indices])
            predictions.append(preds)
            print(f"Predictions from estimator {count} obtained")
            count += 1
        predictions = np.array(predictions)
        def majority_vote(preds):
            counts = np.bincount(preds)
            return np.argmax(counts)
        final = np.apply_along_axis(majority_vote, axis = 0, arr = predictions)
        return final
    
class PerceptronClassifier():
    def __init__(self, learning_rate = 0.01, n_iterations = 1000, n_classes = 10):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_classes = n_classes
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((self.n_classes, n_features))
        self.bias = np.zeros(self.n_classes)
        for _ in range(self.n_iterations):
            for i in range(n_samples):
                x_i = X[i, :]
                y_i = y[i]
                z_i = self.weights @ x_i + self.bias
                y_pred_i = np.argmax(z_i)
                
                if y_pred_i != y_i:
                    self.weights[y_i, :] += self.learning_rate * x_i
                    self.bias[y_i] += self.learning_rate
                    self.weights[y_pred_i, :] -= self.learning_rate * x_i
                    self.bias[y_pred_i] -= self.learning_rate
  
    def predict(self, X):
        Z = X @ self.weights.T + self.bias
        return np.argmax(Z, axis = 1)

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

####################################################################
# Softmax Regression


# print("Softmax Base Prediction:")

# start_time = time.perf_counter()

# model = SoftMaxRegression(learning_rate = 0.001, n_iterations = 50000)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_val)

# end_time = time.perf_counter()

# print("f1_score: ", f1_score(y_val, y_pred))
# print(f"Time taken for training and prediction: {end_time - start_time} seconds")

# accuracy = np.sum(y_pred == y_val) / len(y_val)
# print(f"Val Accuracy: {accuracy}")
###################################
#Bagged Softmax



# print("\nBagged Softmax Predictions:")
# start_time = time.perf_counter()

# base_model = lambda: SoftMaxRegression(learning_rate = 0.01, n_iterations = 10000)
# bagged_model = BaggedClassifier(base_model, n_estimators = 4, subsample_ratio = 0.8)
# bagged_model.fit(X_train, y_train)
# y_pred = bagged_model.predict(X_val)

# end_time = time.perf_counter()

# print("f1_score: ", f1_score(y_val, y_pred))
# print(f"Time Taken: {end_time - start_time} seconds")

# accuracy = np.sum(y_pred == y_val) / len(y_val)
# print(f"Val Accuracy: {accuracy}")

##################################
# Perceptron Classifier


# print("\n Perceptron Base Prediction:")

# start_time = time.perf_counter()

# model = PerceptronClassifier(learning_rate = 0.001  , n_iterations = 10000)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_val)

# end_time = time.perf_counter()

# print("f1_score: ", f1_score(y_val, y_pred))
# print(f"Time Taken: {end_time - start_time} seconds")

# accuracy = np.sum(y_pred == y_val) / len(y_val)
# print(f"Val Accuracy: {accuracy}")

################################################
#KNN Classifier

# print("\n KNN Prediction:")
# start_time = time.perf_counter()
# model = KNNClassifier(n_neighbors = 5)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_val)
# end_time = time.perf_counter()

# print("f1_score: ", f1_score(y_val, y_pred))
# print(f"Time Taken: {end_time - start_time} seconds")
# accuracy = np.sum(y_pred == y_val) / len(y_val)
# print(f"Val Accuracy: {accuracy}")

#####################################################
#Bagged KNN Classifier
# print("\n Bagged KNN Prediction:")
# start_time = time.perf_counter()
# base_model = lambda: KNNClassifier(n_neighbors = 5)
# bagged_model = BaggedClassifier(base_model, n_estimators = 10, subsample_ratio = 0.8, subsample_feature_ratio = 0.8)
# bagged_model.fit(X_train, y_train)
# y_pred = bagged_model.predict(X_val)
# end_time = time.perf_counter()

# print("f1_score: ", f1_score(y_val, y_pred))
# print(f"Time Taken: {end_time - start_time} seconds")
# accuracy = np.sum(y_pred == y_val) / len(y_val)
# print(f"Val Accuracy: {accuracy}")

########################################################################

# KNN works the best and I would like to use it in main.py because of its simplicity and accuracy.
# I have commented all functions out in here for ease of running. I will only be using KNN in main.py.