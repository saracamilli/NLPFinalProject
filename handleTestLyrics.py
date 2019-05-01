# Name: Palmer Robins & Sara Camili
from __future__ import division
from math import log

# THIS FILE HANDLES UNSEEN LYRICS

from GenerateFeatureVectors import nGramCounts, computeProb
from math import log
import random

#################################################################################################################
# Given a new tester sentence, this helper function uses the probability dictionaries to compute the probability
# of a given sentence being from a country and hip hop song. Does this by going through each bi/unigram in the
# line and computing a probability of that bi/unigram being in each category using the Katz-Backoff probabilities
# in the dictionary
#################################################################################################################
def calculateSongProbability_LANG_MODEL(testingEntries, country_lyrics, hiphop_lyrics):

    n = 2

    # Get nGram counts for country training data
    country_nGramCounts = nGramCounts(country_lyrics, n)
    country_nMinus1GramCounts = nGramCounts(country_lyrics, n - 1)
    # Get nGram counts for hip-hop training data
    hiphop_nGramCounts = nGramCounts(hiphop_lyrics, n)
    hiphop_nMinus1GramCounts = nGramCounts(hiphop_lyrics, n - 1)

    # Get total and estimated word counts using helper function
    wordCounts = getWordCounts(country_lyrics, hiphop_lyrics, country_nGramCounts, country_nMinus1GramCounts, \
        hiphop_nGramCounts, hiphop_nMinus1GramCounts)
    country_TotalWordCount = wordCounts[0]
    hiphop_TotalWordCount = wordCounts[1]
    country_estimatedUnknownWordCount = wordCounts[2]
    hiphop_estimatedUnknownWordCount = wordCounts[3]

    results = []    # Stores newly classified test sentences

    counter = 0
    for entry in testingEntries:
        if counter > 30000:
            break
        # Probability that any given sentence is either country or hip-hop
        print(len(country_lyrics))
        print(len(hiphop_lyrics))
        countryProb = -log(len(country_lyrics) / (len(country_lyrics) + len(hiphop_lyrics)))
        hiphopProb = -log(len(hiphop_lyrics) / (len(country_lyrics) + len(hiphop_lyrics)))

        lyric = entry[3:]
        words = lyric.split()

        for i in range(0, len(words) - 2):
            nGram = words[i] + " " + words[i + 1]
            history = words[i].strip()
            countryProb += computeProb(nGram, country_nGramCounts.get(nGram), country_nMinus1GramCounts.get(history),
                                       country_TotalWordCount, country_estimatedUnknownWordCount)
            hiphopProb += computeProb(nGram, hiphop_nGramCounts.get(nGram), hiphop_nMinus1GramCounts.get(history),
                                      hiphop_TotalWordCount, hiphop_estimatedUnknownWordCount)

        # Extract and use keyword features; add 0.05 to probability for every matching keyword in the
        # corresponding genre
        keywordFeat = extractKeywordFeatures(words)
        for i in keywordFeat[0]:
            if i == 1:
                hiphopProb = hiphopProb + 0.4
        for i in keywordFeat[1]:
            if i == 1:
                countryProb = countryProb + 0.4

        if (countryProb > hiphopProb):
            results.append("c: " + lyric)
        elif (hiphopProb > countryProb):
            results.append("h: " + lyric)
        else:
            if (random.random() > 0.5):
                results.append("c: " + lyric)
            else:
                results.append("h: " + lyric)
        counter += 1

    return(results)

#################################################################################################################
# Get the total word counts for both classes, as well as an estimation of unknown word counts
#################################################################################################################
def getWordCounts(country_lyrics, hiphop_lyrics, country_nGramCounts, country_nMinus1GramCounts, \
    hiphop_nGramCounts, hiphop_nMinus1GramCounts):

    country_TotalWordCount = 0
    hiphop_TotalWordCount = 0
    country_estimatedUnknownWordCount = 0
    hiphop_estimatedUnknownWordCount = 0
    for gram, count in country_nMinus1GramCounts.items():
        if count <= 5:
            country_estimatedUnknownWordCount += 1
            #country_nMinus1GramCounts.pop(gram)
        else:
            country_TotalWordCount += count
    for gram, count in hiphop_nMinus1GramCounts.items():
        if count <= 5:
            hiphop_estimatedUnknownWordCount += 1
            #hiphop_nMinus1GramCounts.pop(gram)
        hiphop_TotalWordCount += count

    return(country_TotalWordCount, hiphop_TotalWordCount, country_estimatedUnknownWordCount, \
        hiphop_estimatedUnknownWordCount)

#################################################################################################################
# Given a new tester text, this helper function takes a dictionary of the unigram, bigram, or trigram counts of
# the new text and loops through all the entries. It takes the product of their probabilities,
# and then multiplies them together. It then returns this final probability of the sentence being hip hop or
# country.
#################################################################################################################
def calculateSongProbability_BAYES(song, probDict):

    product = 1

    # Loop through all the words in the sentence
    for word in song.split():

        # Access probabilities of each word in that sentence
        prob = probDict.get(word)

        # If prob is None, change to very small probability
        if (prob is None):
            prob = 0.00000001

        product = product * prob

    return(product)

#################################################################################################################
# Given a text block, extracts presence of keywords for both hip hop and country features and creates two
# feature vectors which it returns as (hip hop vector, country vector)
# TODO: should we change this to frequency, or is presence good enough?
#################################################################################################################
def extractKeywordFeatures(textBlock):

    # Extracted from the Internet - need more hip hop keywords?
    hipHopKeywords = ["chopper", "stunting", "flexing", "mane", "trill", "trapping", "balling" \
        "realest", "homie", "snitch", "biggie", "grind", "nigga", "shit", "bitch", "skrrt", \
            "never", "fuck", "hit", "money", "ass", "big", "real"]

    countryKeywords = ["ride", "baby", "oh", "tobacco", "windows", "blown", \
        "road", "memory", "drunk", "got", "know", "highway", "cold", "beer" \
            "little", "away", "dirt", "town", "chew", "whoa", "plane", "southern", "south" \
                "redneck", "springsteen", "cruise", "truck", "headlights", "town" \
                    "radio", "hey", "rolling", "song", "round", "til", "lane", "wind"]

    # Define and initialize feature vectors with 0's
    hipHopFeatureVect = []
    for i in range(len(hipHopKeywords)): hipHopFeatureVect.append(0)
    countryFeatureVect = []
    for i in range(len(countryKeywords)): countryFeatureVect.append(0)

    # Update the feature vectors
        # NOTE: the words are not filtered because, esp. in the case of hip hop, the
        # suffixes can be meaningful (i.e. balling, not ball)
    for word in textBlock:
        # Check if the current word is in hip hop keywords list
        if word in hipHopKeywords:
            # If it is, replace its 0 in the vector with a 1
            index = hipHopKeywords.index(word)
            hipHopFeatureVect[index] = 1

        # Check if the current word is in country keywords list
        if word in countryKeywords:
            # If it is, replace its 0 in the vector with a 1
            index = countryKeywords.index(word)
            countryFeatureVect[index] = 1

    return(hipHopFeatureVect, countryFeatureVect)