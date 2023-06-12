from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_metrics(actual_labels, predicted_labels):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for actual, predicted in zip(actual_labels, predicted_labels):
        #     true positive (TP) is a case for which both the actual and the predicted
        #     classes are positive;
        if actual == 1 and predicted == 1:
            true_positive += 1
        #      true negative (TN) is a case for which both the actual and the predicted
        #      classes are negative;
        elif actual == 0 and predicted == 0:
            true_negative += 1
        #     false positive (FP) is a case in which the model predicts the positive
        #     class while the actual class is negative;
        elif actual == 0 and predicted == 1:
            false_positive += 1
        #     false negative (FN) is a case in which the model predicts the negative
        #     class while the actual class is positive.
        elif actual == 1 and predicted == 0:
            false_negative += 1
    precision = true_positive / (true_positive + false_positive)
    return true_positive, true_negative, false_positive, false_negative, precision


# example of usage
# ms.plot_confusion_matrix(Y_valid, predictions_validation, ['clickbait','non-clickbait'])
def plot_confusion_matrix(actual_labels, predicted_labels, classes):
    cm = confusion_matrix(actual_labels, predicted_labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title("Confusion Matrix")
    plt.show()
