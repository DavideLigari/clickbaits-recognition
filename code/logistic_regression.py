import numpy as np
import matplotlib.pyplot as plt


def logreg_inference(X, w, b):
    z = X @ w + b
    p = 1 / (1 + np.exp(-z))
    return p


def cross_entropy(P, Y):
    P = np.clip(P, 0.0001, 0.9999)
    return (-Y * np.log(P) - (1 - Y) * np.log(1-P)).mean()


def logreg_train(X, Y, X_valid, Y_valid, steps, lr, lambda_):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    accs_train = []
    accs_valid = []
    for step in range(steps+1):
        P = logreg_inference(X, w, b)
        if step % 1000 == 0:
            P_valid = logreg_inference(X_valid, w, b)
            prediction = (P_valid > 0.5)
            accuracy = (prediction == Y_valid).mean()
            accs_valid.append(accuracy*100)
            print(
                f"step {step}, accuracy {accuracy:.4f}")
            P_train = logreg_inference(X, w, b)
            prediction = (P_train > 0.5)
            accuracy = (prediction == Y).mean()
            accs_train.append(accuracy*100)
        grad_w = ((P - Y) @ X) + 2 * lambda_ * w
        grad_b = (P - Y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, accs_valid, accs_train
