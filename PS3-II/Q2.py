import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def load(digit0=0, digit1=1):
    x_train = np.loadtxt("data-mnist/trainX.txt", delimiter=",").astype(np.float32)
    y_train = np.loadtxt("data-mnist/trainY.txt", delimiter=",").astype(np.float32)

    x_test = np.loadtxt("data-mnist/testX.txt", delimiter=",").astype(np.float32)
    y_test = np.loadtxt("data-mnist/testY.txt", delimiter=",").astype(np.float32)

    # Filter selected digits
    train_mask = np.isin(y_train, [digit0, digit1])
    test_mask = np.isin(y_test, [digit0, digit1])

    x_train = x_train[train_mask]
    x_test = x_test[test_mask]

    # Convert original labels into binary labels
    y_train = (y_train[train_mask] == digit1).astype(int)
    y_test = (y_test[test_mask] == digit1).astype(int)

    # Add bias column
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])
    return x_train, y_train, x_test, y_test
def gradient_of_U(beta, x, y, sigma):
    p = sigmoid(x @ beta)
    gradient = x.T @ (p - y) + beta / sigma**2
    return gradient
def U(beta, x, y, sigma):
    logits = x @ beta
    negative_log_likelihood = np.sum(np.logaddexp(0, logits) - y * logits)
    negative_log_prior = np.dot(beta, beta) / (2 * sigma**2)
    return negative_log_likelihood + negative_log_prior
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))
def map_estimate(x,y,steps,epsilon,sigma):
    beta = np.zeros(x.shape[1])
    betas = []
    for i in range(steps):
        grad = gradient_of_U(beta, x, y, sigma)
        beta = beta - epsilon * grad
        betas.append(beta)
        if i % 500 == 0:
            print(f"Step {i}: U = {U(beta, x, y, sigma):.4f}")
    return beta

x_train, y_train, x_test, y_test = load(6,8)
x_train = x_train[:, 1:]
x_test = x_test[:, 1:]
df = pd.DataFrame(
    x_train,
    columns=[f"pixel_{i}" for i in range(784)]
)


scaler = StandardScaler()
scaled_train = scaler.fit_transform(df)
scaled_test = scaler.transform(x_test)

start = time.perf_counter()
pca = PCA(n_components=10)

principal_components = pca.fit_transform(scaled_train)
test_components = pca.transform(scaled_test)

X_train_pca = np.column_stack([
    np.ones(len(principal_components)),
    principal_components
])

X_test_pca = np.column_stack([
    np.ones(len(test_components)),
    test_components
])

beta = map_estimate(X_train_pca,y_train,5000,5e-5,1.2)
# print(beta)

p_predicted =sigmoid(X_test_pca @beta)
y_pred = (p_predicted >= 0.5).astype(int)


error_rate = np.mean(np.abs(y_test - y_pred))
accuracy = 1 - error_rate
elapsed = time.perf_counter() - start

# test()
print(f"PC={pca}, Accuracy={accuracy:.2%}, Time={elapsed:.4f}s")