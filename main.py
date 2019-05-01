
# Name: Palmer Robins & Sara Camili

import csv
import random

from Parser import formatText, readCSVFile
from GenerateFeatureVectors import computeProb, nGramCounts
from handleTestLyrics import calculateSongProbability_LANG_MODEL, calculateSongProbability_BAYES, \
    extractKeywordFeatures
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

    # TESTING
    results = calculateSongProbability_LANG_MODEL(testingEntries, country_lyrics, hiphop_lyrics)
    printStatistics(results, testingEntries)

###############################################################################################################
if __name__ == "__main__":
    # execute only if run as a script
    main()