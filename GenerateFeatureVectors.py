# Name: Palmer Robins & Sara Camili

import nltk

import codecs
from nltk.tokenize import PunktSentenceTokenizer,sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
# nltk.download('wordnet')

#################################################################################################################
# Given lyrical sentences in both hip hop and country, generate bigram and unigram probability dictionaries
# which will be used to classify new tester data
# Input: hip hop lyrical sentences, country lyrical sentences in list form
#################################################################################################################
def generateTrainingDicts_LanguageModels(hipHopData, countryData):

    hipHopBigramProbDict = createBigramProbabilityDict(hipHopData)
    hipHopUnigramProbDict = createUnigramProbabilityDict(hipHopData)
    countryBigramProbDict = createBigramProbabilityDict(countryData)
    countryUnigramProbDict = createUnigramProbabilityDict(countryData)

    return (hipHopBigramProbDict, hipHopUnigramProbDict, countryBigramProbDict, countryUnigramProbDict)

#################################################################################################################
# Computes unigram and bigram Bayes probability dictionary for the training lyrical text
# Input: a block of training lyrical sentences from either country or hip hop, and total counts of unigrams, bigrams,
# and trigrams in training text
# Output: unigram, bigram, and trigram dictionaries with the Bayes probability of each unigram, bigram, or trigram
#################################################################################################################
def generateTrainingDicts_Bayes(hipHopData, countryData):

    # Get unigram counts of both
    hipHopCounts = unigramCounts(hipHopData)
    countryCounts = unigramCounts(countryData)

    # Get the number of unigrams in the text
    numberOfUnigrams = len(hipHopCounts.items()) + len(countryCounts.items())

    # Probability for all words in plot sentences
    hipHopUnigramProbabilityDictionary = {}
    countryUnigramProbabilityDictionary = {}

    for word in hipHopCounts.keys():
        prob = hipHopCounts.get(word) / numberOfUnigrams
        hipHopUnigramProbabilityDictionary.update({word : prob})

    # Probability for all words in review sentences
    for word in countryCounts.keys():
        prob = countryCounts.get(word) / numberOfUnigrams
        countryUnigramProbabilityDictionary.update({word : prob})

    return(hipHopUnigramProbabilityDictionary, countryUnigramProbabilityDictionary)

###########################################################################################################
################################# SORTING & PROBABILITY FUNCTIONS #########################################
###########################################################################################################

#################################################################################################################
# Computes all bigrams and their counts in the given text
# Input: training or test text file
# Output: a dictionary of all bigrams and their counts
#################################################################################################################
def bigramCounts(sentences):

    myBigramDict = {}                        # will hold all the bigrams and their counts

    # Loop through and find all the possible bigrams, adding them
    # to its dictionary. If it already exists in the dictionary, update its count
    counter = 0
    for sentence in sentences:              # loop through all sentences in input
        sentence = sentence.split()         # split sentence into words
        for word in sentence:               # loop through all words in sentence
            new = {}                            # will hold new dictionary entry
            currBigram = []                      # will be used to hold the current bigram
                                                    # (under construction)
            currBigramStr = ""                   # will hold the converted string version of the bigram

            currBigram.append(word)

            if (counter + 1 < len(sentence)):         # if we haven't reached the end of the file...
                nextWord = sentence[counter+1]
                currBigram.append(nextWord)             # add the next word to the bigram
                currBigramStr = ' '.join(str(w) for w in currBigram)            # convert to a string
                # currBigramStr.replace("[","").replace("]","").replace("'","")       # get rid of brackets and ' '

                if (not(myBigramDict.get(currBigramStr) is None)):
                    count = myBigramDict[currBigramStr] + 1      # find it's current count
                    myBigramDict[currBigramStr] = count          # update the dictionary

                else:
                    new = {currBigramStr : 1}                    # otherwise, add new entry with count 1
                    myBigramDict.update(new)                     # update the dictionary

                counter = counter + 1

    # Return this dictionary with all bigrams and their counts
    return(myBigramDict)

#################################################################################################################
# Computes all unigrams and their counts in the given text
# Input: training or test text file
# Output: a dictionary of all unigrams and their counts
#################################################################################################################
def unigramCounts(sentences):

    myUnigramDict = {}                              # will hold the unigrams and all their counts

    # Loop through all the words in the file and add unigrams to dictionary. If a unigram
    # already exists in the dictionary, update its count
    for sentence in sentences:
        sentence = sentence.split()
        for word in sentence:
            # unigramString = word.replace("[","").replace("]","").replace("'","")    # get rid of brackets and junk
            unigramString = word

            if (myUnigramDict.get(unigramString) is None):     # if not already in the dictionary...
                myUnigramDict.update({unigramString : 1})           # add to dictionary with a count of 1

            else:
                count = myUnigramDict.get(unigramString) + 1            # get current count
                myUnigramDict[unigramString] = count                # update it with one more than that

    return(myUnigramDict)

#################################################################################################################
# Creates and returns a dictionary of probabilities for all bigrams in the input text. This dictionary will be
# of the format {bigram : probability} and will be referenced when a new text is encountered. Will ultimately
# be called twice per file (once with all plot sentences and once with all review sentences)
#################################################################################################################
def createBigramProbabilityDict(sentences):

    probDictionary = {}             # the dictionary to be returned

    bigramCountDictionary = bigramCounts(sentences)         # generate the bigram count dictionary
    unigramCountDictionary = unigramCounts(sentences)       # generate the unigram count dictionary
    vocabSize = len(unigramCountDictionary.items())         # get vocab size

    # Loop through all bigrams, access count, and generate Katz-backoff probability
    for bigram in bigramCountDictionary.keys():
        bigramCount = bigramCountDictionary.get(bigram)

        # Get the count of this history (unigram) by separating out history from token
        bigram = str(bigram).lstrip().rstrip()                  # remove leading and training whitespace
        splitBigram = bigram.split()                       # split around the blank between history & token
        unigram = splitBigram[0]                           # access just the history
        unigramCount = unigramCountDictionary.get(unigram)      # get count of this history

        prob = generateKatzBackoffProbability(bigram, bigramCount, unigram, unigramCount, vocabSize)
        probDictionary.update({bigram : prob})

    return(probDictionary)

#################################################################################################################
# Creates and returns a dictionary of probabilities for all unigrams in the input text. This dictionary will be
# of the format {unigram : probability} and will be referenced when a new text is encountered. Will ultimately
# be called twice per file (once with all plot sentences and once with all review sentences)
#################################################################################################################
def createUnigramProbabilityDict(sentences):
    probDictionary = {}             # the dictionary to be returned

    unigramCountDictionary = unigramCounts(sentences)       # generate the unigram count dictionary
    vocabSize = len(unigramCountDictionary.items())         # get vocab size

    # Loop through all unigrams, access count, and generate Katz-backoff probability
    for unigram in unigramCountDictionary.keys():
        unigramCount = unigramCountDictionary.get(unigram)
        unigram = str(unigram).lstrip().rstrip()                  # remove leading and training whitespace

        prob = generateKatzBackoffProbability(None, None, unigram, unigramCount, vocabSize)
        probDictionary.update({unigram : prob})

    return(probDictionary)

#################################################################################################################
 # Given counts of all the bigrams and unigrams, along with the overall size of the vocabulary in the text,
 # generate probabilities using Katz-backoff with absolute discounting
 # TODO: this function needs to be fixed! Had some problems
 #################################################################################################################
def generateKatzBackoffProbability(bigram, bigramCount, unigram, unigramCount, vocabSize):
    D = 0.75            # Let D be 0.75, the standard value
    probability = 0     # Will hold the probability to be returned

    # We've backed off one time, so we're evaluating the Katz probability of a unigram
    if (bigram == None):
        # We've never seen this unigram before
        if (unigramCount == 0):
            probability = 0         # FIX THIS, to use open vocab! Keep a count of unknowns and then use that prob

        # We've seen this unigram before: P* = C(a)-d / V, where V is size of vocab
        else:
            probability = (unigramCount - D) / vocabSize

        return probability

    # If the count of this bigram is 0...
    if (bigramCount == 0):

        # Calculate P* = C(a b)-d / c(a)
        Pstar = (bigramCount - D) / unigramCount

        # Calculate Pkatz(a)
        PkatzUnigram = generateKatzBackoffProbability(None, None, unigram, unigramCount, vocabSize)

        # Calculate alpha = (1 - P*(a b)) / Pkatz(a)
        alpha = (1 - Pstar) / PkatzUnigram

        # Calculate probability = alpha * Pkatz(a)
        probability = alpha * PkatzUnigram


    # We've seen this bigram before
    else:
        try:
            probability = (bigramCount - D) / unigramCount
        except TypeError as e:
            print("Bigram: " + bigram + "\n")
            print("Bigram count: " + str(bigramCount) + "\n")
            print("Unigram: " + unigram + "\n")
            print("Unigram count: " + str(unigramCount) + "\n")
            exit(1)

    return probability

