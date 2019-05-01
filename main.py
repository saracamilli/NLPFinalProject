# Name: Palmer Robins & Sara Camili
from __future__ import division
from math import log

import csv
import random

from Parser import formatText, readCSVFile
from GenerateFeatureVectors import computeProb, nGramCounts
from statistics import printStatistics

def main():
    n = 2
    country_lyrics = []     # Stores all testing country lyrics
    hiphop_lyrics = []      # Stores all testing hip-hop lyrics

    # Store country/hip-hop lyrics from the csv as sentences in list
    entries = readCSVFile("lyrics.csv")

    # use half the list for training, half for testing
    trainingEntries = entries[:len(entries)//2]
    testingEntries = entries[len(entries)//2:]

    # Use the genre label to insert each lyric into the country or hip-hop dataset
    print("Classifying training data...")
    for lyric in trainingEntries:
        if (lyric[0] == 'c'):
            lyric = lyric[3:]
            country_lyrics.append(lyric)
        else:
            lyric = lyric[3:]
            hiphop_lyrics.append(lyric)
    print("Done classifying training data!")

    # Format the text
    print("Formatting country lyrics...")
    country_lyrics = formatText(country_lyrics)
    print("Done formatting country lyrics!\nFormatting hip-hop lyrics...")
    hiphop_lyrics = formatText(hiphop_lyrics)
    print("Done formatting hip-hop lyrics!")

    # Get nGram counts for country training data
    country_nGramCounts = nGramCounts(country_lyrics, n)
    country_nMinus1GramCounts = nGramCounts(country_lyrics, n - 1)
    # Get nGram counts for hip-hop training data
    hiphop_nGramCounts = nGramCounts(hiphop_lyrics, n)
    hiphop_nMinus1GramCounts = nGramCounts(hiphop_lyrics, n - 1)

    # Get the total word counts for both classes, as well as an estimation of unknown word counts
    country_TotalWordCount = 0
    hiphop_TotalWordCount = 0
    country_estimatedUnknownWordCount = 0
    hiphop_estimatedUnknownWordCount = 0
    for gram, count in country_nMinus1GramCounts.items():
        if count <= 5:
            country_estimatedUnknownWordCount += 1
            country_nMinus1GramCounts.pop(gram)
        else:
            country_TotalWordCount += count
    for gram, count in hiphop_nMinus1GramCounts.items():
        if count <= 5:
            hiphop_estimatedUnknownWordCount += 1
            hiphop_nMinus1GramCounts.pop(gram)
        hiphop_TotalWordCount += count

    # TESTING
    results = []    # Stores newly classified test sentences
    counter = 0
    print("Formatting Testing Entries...")
    testingEntries = formatText(testingEntries)
    print("Done formatting testing entries!")
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
            countryProb += computeProb(nGram, country_nGramCounts.get(nGram), country_nMinus1GramCounts.get(history),
                                       country_TotalWordCount, country_estimatedUnknownWordCount)
            hiphopProb += computeProb(nGram, hiphop_nGramCounts.get(nGram), hiphop_nMinus1GramCounts.get(history),
                                      hiphop_TotalWordCount, hiphop_estimatedUnknownWordCount)
        if (countryProb > hiphopProb):
            results.append("c: " + lyric)
        elif (hiphopProb > countryProb):
            results.append("h: " + lyric)
        else:
            print("The probabilities are the same")
            if (random.choice(1,-1) > 0):
                results.append("c: " + lyric)
            else:
                results.append("h: " + lyric)
        counter += 1

    printStatistics(results, testingEntries)

###############################################################################################################
if __name__ == "__main__":
    # execute only if run as a script
    main()