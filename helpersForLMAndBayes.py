# Name: Palmer Robins & Sara Camili
from __future__ import division

import nltk
from nltk.corpus import wordnet

from math import log
# nltk.download('wordnet')

#################################################################################################################
# Computes all nGrams and their counts in the given text
# Input: training or test text file
# Output: a dictionary of all nGrams and their counts
#################################################################################################################
def nGramCounts(sentences, n):
    counts = {}

    # For each sentence in the set of sentences
    for words in sentences:
        # Construct the entries, count the entries, and place in the dictionary
        for i in range(len(words)):
            word = words[i]
            for j in range(1, n):
                if i + j < len(words):
                    word = word + " " + words[j + i]
                else:
                    break

            if n == 2:
                word = word.strip()
                lengthChecker = len(word.split())
                if len(word) == 0:
                    continue
                if lengthChecker < 2:
                    continue


            if (counts.get(word) is None):
                counts.update({word : 1})
            else:
                counts[word] = counts.get(word) + 1

    return counts                                   # Return this dictionary with all bigrams and their counts

#################################################################################################################
 # Given counts of all the bigrams and unigrams, along with the overall size of the vocabulary in the text,
 # generate probabilities using Katz-backoff with absolute discounting
 #################################################################################################################
def computeProb_LM(entry, nGramCount, nMinus1GramCount, vocabSize, unknownWordCount):
    words = entry.split()   # Split the n-gram into words
    history = words[0]

    if nGramCount is None or nMinus1GramCount is None:
        if (nMinus1GramCount is None):
            prob = -log(unknownWordCount / vocabSize)
            return prob
        else:
            prob = -log(nMinus1GramCount / vocabSize)
            return prob

    try:
        return -log((nGramCount - 0.75) / nMinus1GramCount)
    except TypeError:
        print(entry)
        print(nGramCount)
        print(history)
        print (nMinus1GramCount)

#################################################################################################################
 # Given counts of all unigrams, along with a given word, generates a probability using Naive Bayes
 #################################################################################################################
def computeProb_Bayes(word, unigramCounts, unkWordCount):
    if (unigramCounts.get(word) is None):
        return -log(unkWordCount / len(unigramCounts))
    else:
        wordCount = unigramCounts.get(word)
        return -log(wordCount / len(unigramCounts))
