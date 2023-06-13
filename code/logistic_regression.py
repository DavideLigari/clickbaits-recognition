import numpy as np


def logreg_inference(X, w, b):
    """Return the probability of being positive."""
    z = X @ w + b
    p = 1 / (1 + np.exp(-z))
    return p


def cross_entropy(P, Y):
    """Return the cross-entropy loss."""
    P = np.clip(P, 0.0001, 0.9999)
    return (-Y * np.log(P) - (1 - Y) * np.log(1-P)).mean()


def logreg_train(X, Y, X_valid, Y_valid, steps=10000, lr=0.01, batch_size=2000):
    """Train a logistic regression classifier."""
    validation_accuracy = []
    train_accuracy = []
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    for step in range(steps+1):
        index = np.random.choice(X.shape[0], batch_size, replace=False)
        newX = X[index]
        newY = Y[index]

        P = logreg_inference(newX, w, b)
        grad_w = ((P - newY) @ newX) / batch_size
        grad_b = (P - newY).mean()

        w -= lr * grad_w
        b -= lr * grad_b

        if step % 1000 == 0:
            P = logreg_inference(X, w, b)
            predictions_train = (P > 0.5).astype(int)
            accuracy_train = (predictions_train == Y).mean()*100
            train_accuracy.append(accuracy_train)

            P = logreg_inference(X_valid, w, b)
            predictions_valid = (P > 0.5).astype(int)
            accuracy_valid = (predictions_valid == Y_valid).mean()*100
            validation_accuracy.append(accuracy_valid)
            print(
                f"Step {step}, train accuracy: {accuracy_train:.2f}%, validation accuracy: {accuracy_valid:.2f}%")

    return w, b, train_accuracy, validation_accuracy
