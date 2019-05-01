from __future__ import division
from __future__ import print_function

import nltk
from nltk.metrics import *

import sys
import random

# This is the program that, given a new text, it processes it and runs both the bigram classifiers and the naive
# Bayes classifer. Given their classifications, it generates statistics to evaluate and compare the approaches.
def printStatistics(results, testingEntries):
    c_truePos = 0
    c_falsePos = 0
    c_trueNeg = 0
    c_falseNeg = 0

    h_truePos = 0
    h_falsePos = 0
    h_trueNeg = 0
    h_falseNeg = 0

    # Country is the positive case
    counter = 0
    for entry in testingEntries:
        if counter >= 30000:
            break
        print(len(results))
        print(counter)
        print("Testing Entry With Label: " + entry)
        if (entry[0] == 'c'):
            print("Testing Entry Recognized as Country")
            print("Results Entry Should Match: " + results[counter])
            if (results[counter][0] == 'c'):
                print("Correctly Classified as Country")
                c_truePos += 1
                h_trueNeg += 1
            else:
                print("Incorrectly Classified as Hip-Hop")
                c_falseNeg += 1
                h_falsePos += 1
        else:
            print("Testing Entry Recognized as Hip-Hop")
            print("Results Entry Should Match: " + results[counter])
            if (results[counter][0] == 'h'):
                print("Correctly Classified as Hip-Hop")
                h_truePos += 1
                c_trueNeg += 1
            else:
                print("Incorrectly Classified as Country")
                h_falseNeg += 1
                c_falsePos += 1
        counter += 1

    # Print the Result Measures
    print("Country True Pos: " + str(c_truePos))
    print("Country True Neg: " + str(c_trueNeg))
    print("Country False Pos: " + str(c_falseNeg))
    print("Country False Neg: " + str(c_falseNeg))
    accuracy = (c_truePos + c_trueNeg) / (c_trueNeg + c_truePos + c_falseNeg + c_falsePos)
    print("ACCURACY: " + str(accuracy))

    precision = reportPrecision(c_truePos, c_trueNeg)
    print("PRECISION: " + str(precision))

    recall = reportRecall(c_truePos, c_falseNeg)
    print("RECALL: " + str(recall))

    f1Measure = reportF1(precision, recall)
    print("F1 MEASURE: " + str(f1Measure))



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
    except Exception:
        denominator = 0.000000000001
        f1 = numerator / denominator
    return f1