from pathlib import Path

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

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


def test():
    logits = x_test @ beta
    p_hat = expit(logits)

    print("Logits:", logits.min(), logits.max())
    print("Probabilities:", p_hat.min(), p_hat.max())

    print("Mean probability for class 0:",
          p_hat[y_test == 0].mean())

    print("Mean probability for class 1:",
          p_hat[y_test == 1].mean())
    train_pred = (expit(x_train @ beta) >= 0.5).astype(int)
    test_pred = (expit(x_test @ beta) >= 0.5).astype(int)

    print("Training accuracy:", np.mean(train_pred == y_train))
    print("Testing accuracy:", np.mean(test_pred == y_test))

    print("Training predictions:",
          np.unique(train_pred, return_counts=True))
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

    if beta.size == 785:
        beta_image = beta[1:].reshape(28, 28)
    elif beta.size == 784:
        beta_image = beta.reshape(28, 28)
    l =np.asarray(betas)

    print("Beta range:", l.min(), l.max())
    plt.figure(figsize=(6, 5))
    plt.imshow(beta_image, cmap="bwr")
    plt.colorbar(label="Beta coefficient")
    plt.title("MAP Logistic Regression Coefficients")
    plt.axis("off")
    plt.show()
    return beta

x_train, y_train, x_test, y_test = load(6,8)
beta = map_estimate(x_train,y_train,5000,5e-4,1.2)
# print(beta)
p_predicted =sigmoid(x_test @ beta)
y_pred = (p_predicted >= 0.5).astype(int)


error_rate = np.mean(np.abs(y_test - y_pred))
accuracy = 1 - error_rate

# test()
print("Training labels:", np.unique(y_train, return_counts=True))
print("Test labels:", np.unique(y_test, return_counts=True))
print("Predicted labels:", y_pred)
print("True labels:     ", y_test)
print(f"Error rate: {error_rate:.4f}")
print(f"Accuracy: {accuracy:.2%}")
