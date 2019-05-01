# Name: Palmer Robins & Sara Camili
from math import log

# THIS FILE HANDLES UNSEEN LYRICS

from GenerateFeatureVectors import nGramCounts, computeProb_LM, computeProb_Bayes
from math import log
import random

#################################################################################################################
# Given a new tester sentence, this helper function uses the probability dictionaries to compute the probability
# of a given sentence being from a country and hip hop song. Does this by going through each bi/unigram in the
# line and computing a probability of that bi/unigram being in each category using the Katz-Backoff probabilities
# in the dictionary
#################################################################################################################
def calculateSongProbability_LANG_MODEL(testingEntries, country_lyrics, hiphop_lyrics):

    n = 2   # bigram model

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
        countryProb = -log(len(country_lyrics) / (len(country_lyrics) + len(hiphop_lyrics)))
        hiphopProb = -log(len(hiphop_lyrics) / (len(country_lyrics) + len(hiphop_lyrics)))

        lyric = entry[3:]
        words = lyric.split()

        for i in range(0, len(words) - 2):
            nGram = words[i] + " " + words[i + 1]
            history = words[i].strip()
            countryProb += computeProb_LM(nGram, country_nGramCounts.get(nGram), country_nMinus1GramCounts.get(history),
                                       country_TotalWordCount, country_estimatedUnknownWordCount)
            hiphopProb += computeProb_LM(nGram, hiphop_nGramCounts.get(nGram), hiphop_nMinus1GramCounts.get(history),
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

        # Incorporate Naive Bayes features as well
        #bayes_probs = calculateSongProbability_BAYES(testingEntries, country_lyrics, hiphop_lyrics)
        #countryProb = countryProb + bayes_probs[0]
        #hiphopProb = hiphopProb + bayes_probs[1]

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
    for count in country_nMinus1GramCounts.values():
        if count <= 5:
            country_estimatedUnknownWordCount += 1
        else:
            country_TotalWordCount += count
    for count in hiphop_nMinus1GramCounts.values():
        if count <= 5:
            hiphop_estimatedUnknownWordCount += 1
        hiphop_TotalWordCount += count

    return(country_TotalWordCount, hiphop_TotalWordCount, country_estimatedUnknownWordCount, \
        hiphop_estimatedUnknownWordCount)

#################################################################################################################
# Given counts of all the unigrams, along with the overall size of the vocabulary in the text,
# generate probabilities using Naive Bayes with unigram features..
#################################################################################################################
def calculateSongProbability_BAYES(testingEntries, country_lyrics, hiphop_lyrics):

    # Get unigram counts of each category 
    hiphop_unigramCounts = nGramCounts(country_lyrics, 1)
    country_unigramCounts = nGramCounts(hiphop_lyrics, 1)

    # Loop through all lyrical entries
    for entry in testingEntries:

        hiphopProb = 0
        countryProb = 0
        # Loop through all the words in the sentence
        for word in entry.split():

            # Compute probabilities using helper function
            hiphopProb = hiphopProb + computeProb_Bayes(word, True, hiphop_unigramCounts, country_unigramCounts)
            countryProb = countryProb + computeProb_Bayes(word, False, hiphop_unigramCounts, country_unigramCounts)
        
        # Invert the log probabilities
        countryProb = - countryProb
        hiphopProb = - hiphopProb

    return(countryProb, hiphopProb)

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