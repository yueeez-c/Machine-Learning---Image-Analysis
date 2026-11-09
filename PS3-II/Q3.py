from pathlib import Path

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import cv2
import numpy as np
from scipy.ndimage import rotate, shift

def augment_image(image):
    theta = np.random.uniform(-15, 15)
    tx = np.random.uniform(-5, 5)
    ty = np.random.uniform(-5, 5)

    # Rotation
    rotated = rotate(image, angle=theta, reshape=False, order=1,  mode="constant", cval=0)
    # Translation
    translated = shift(rotated, shift=(ty, tx), order=1,  mode="constant",  cval=0)
    return translated
def augment_flattened(image):
    side = int(np.sqrt(image.size))

    if side * side != image.size:
        raise ValueError("Expected a flattened square image")

    image_2d = image.reshape(side, side)

    augmented = augment_image(image_2d)

    return augmented.flatten()
def augment_dataset(X, Y, times):
    X_aug = []
    Y_aug = []
    for image, label in zip(X, Y):
        # Keep original image
        X_aug.append(image)
        Y_aug.append(label)
        # Generate additional images
        for _ in range(times - 1):
            X_aug.append(augment_flattened(image))
            Y_aug.append(label)

    return np.array(X_aug), np.array(Y_aug)
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
        # if i % 500 == 0:
        #     print(f"Step {i}: U = {U(beta, x, y, sigma):.4f}")

    if beta.size == 785:
        beta_image = beta[1:].reshape(28, 28)
    elif beta.size == 784:
        beta_image = beta.reshape(28, 28)
    l =np.asarray(betas)

    # print("Beta range:", l.min(), l.max())
    # plt.figure(figsize=(6, 5))
    # plt.imshow(beta_image, cmap="bwr")
    # plt.colorbar(label="Beta coefficient")
    # plt.title("MAP Logistic Regression Coefficients")
    # plt.axis("off")
    # plt.show()
    return beta

def add_bias_column(a):
    return np.column_stack([np.ones(len(a)), a])
def result(x, beta):
    p_predicted = sigmoid(x @ beta)
    y_pred = (p_predicted >= 0.5).astype(int)
    error_rate = np.mean(np.abs(y_test - y_pred))
    accuracy = 1 - error_rate
    # print("Predicted labels:", y_pred)
    # print("True labels:     ", y_test)
    print(f"Error rate: {error_rate:.4f}")
    print(f"Accuracy: {accuracy:.2%}")

x_train, y_train, x_test, y_test = load(6,8)
# Remove bias column
x_train = x_train[:, 1:]

#random seed
rng = np.random.default_rng(42)
selected = []
for label in [0, 1]:
    indices = np.where(y_train == label)[0]

    # Select 10% of this class
    n = round(len(indices) * 0.1)
    chosen = rng.choice(indices, size=n, replace=False)
    selected.extend(chosen)

x_small = x_train[selected]
y_small = y_train[selected]

X_3x, Y_3x = augment_dataset(x_small, y_small, 3)
X_5x, Y_5x = augment_dataset(x_small, y_small, 5)

X_3x = add_bias_column(X_3x)
X_5x = add_bias_column(X_5x)
x_small = add_bias_column(x_small)
print("original")
beta = map_estimate(x_small,y_small,5000,5e-4,1.2)
result(x_test,beta)

print("3x")
beta = map_estimate(X_3x,Y_3x,5000,5e-4,1.2)
result(x_test,beta)

print("5x")
beta = map_estimate(X_5x,Y_5x,5000,5e-4,1.2)
result(x_test,beta)
