# Name: Palmer Robins & Sara Camili
from __future__ import division
import csv

from math import log

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from Parser import readCSVFile
from GenerateFeatureVectors import computeProb, nGramCounts
from statistics import printStatistics

import random

def main():

    country_lyrics = []     # Stores all testing country lyrics
    hiphop_lyrics = []      # Stores all testing hip-hop lyrics
    n = 2
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
    print("Hip-Hop lyric count: " + str(len(hiphop_lyrics)))
    print("Country lyric count: " + str(len(country_lyrics)))

    # Format the text
    print("Formatting country lyrics...")
    country_lyrics = formatText(country_lyrics)
    print("Done formatting country lyrics!\nFormatting hip-hop lyrics...")


    hiphop_lyrics = formatText(hiphop_lyrics)
    print("Done formatting hip-hop lyrics!")

    country_nGramCounts = nGramCounts(country_lyrics, n)
    country_nMinus1GramCounts = nGramCounts(country_lyrics, n - 1)

    hiphop_nGramCounts = nGramCounts(hiphop_lyrics, n)
    hiphop_nMinus1GramCounts = nGramCounts(hiphop_lyrics, n - 1)

    country_TotalWordCount = 0
    hiphop_TotalWordCount = 0
    country_estimatedUnknownWordCount = 0
    hiphop_estimatedUnknownWordCount = 0

    # Get the total word counts for both classes, as well as an estimation of unknown word counts
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
    results = []

    print("Number of Testing Entries: " + str(len(testingEntries)))

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
            countryProb += computeProb(nGram, country_nGramCounts.get(nGram), country_nMinus1GramCounts.get(history),
                                       country_TotalWordCount, country_estimatedUnknownWordCount)
            hiphopProb += computeProb(nGram, hiphop_nGramCounts.get(nGram), hiphop_nMinus1GramCounts.get(history),
                                      hiphop_TotalWordCount, hiphop_estimatedUnknownWordCount)
        if (countryProb > hiphopProb):
            results.append(entry)
        elif (hiphopProb > countryProb):
            results.append(entry)
        else:
            if (random.choice(1,-1) > 0):
                results.append(entry)
            else:
                results.append(entry)
        counter += 1

    printStatistics(results, testingEntries)




def formatText(lyrics):
    formattedLyrics = []
    porter = PorterStemmer()
    for lyric in lyrics:
        if lyrics.index(lyric) > 10000:
            break
        try:
            tokens = word_tokenize(lyric)
            words = [word for word in tokens if word.isalpha()]
            stemmed = ' '.join([porter.stem(word) for word in tokens])
            formattedLyrics.append(stemmed)
        except UnicodeDecodeError as e:
            continue
    return formattedLyrics


###############################################################################################################
if __name__ == "__main__":
    # execute only if run as a script
    main()