import sys
import random

# This is the program that, given a new text, it processes it and runs both the bigram classifiers and the naive
# Bayes classifer. Given their classifications, it generates statistics to evaluate and compare the approaches.

# Given counts of true positives and false positives, calculates and returns a precision value
def reportPrecision(truePositives, falsePositives):
    precision = truePositives / (truePositives + falsePositives)
    return(precision)


# Given counts of true positives and false negatives, calculates and returns a recall value
def reportRecall(truePositives, falseNegatives):
    recall = truePositives / (truePositives + falseNegatives)
    return(recall)


# Given precision and recall values, calculates and returns an F1 measure
def reportF1(precision, recall):
    numerator = 2 * precision * recall
    denominator = precision + recall

    # Avoid a divide by 0 error if both precision and recall are 0
    try:
        f1 = numerator / denominator
        pass
    except Exception as e:
        denominator = 0.000000000001
        f1 = numerator / denominator
    return(f1)