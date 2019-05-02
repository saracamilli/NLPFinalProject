
# Name: Palmer Robins & Sara Camili

import csv

from Parser import formatText, readCSVFile
from handleTestLyrics import calculateTestingProbabilities
from statistics import printStatistics

def main():
    country_lyrics = []     # Stores all testing country lyrics
    hiphop_lyrics = []      # Stores all testing hip-hop lyrics

    # Store country/hip-hop lyrics from the csv as sentences in list
    print("Reading CSV input file...")
    entries = readCSVFile("lyrics.csv")
    entries = list(dict.fromkeys(entries))
    print("Done reading CSV input file!")

    # use half the list for training, half for testing
    trainingEntries = entries[:len(entries)//2]
    testingEntries = entries[len(entries)//2:]

    # Use the genre label to insert each lyric into the country or hip-hop dataset
    print("Labeling training data...")
    for lyric in trainingEntries:
        if (lyric[0] == 'c'):
            lyric = lyric[3:]
            country_lyrics.append(lyric)
        else:
            lyric = lyric[3:]
            hiphop_lyrics.append(lyric)
    print("Done labeling training data!")

    # Format the text
    print("Formatting country data...")
    country_lyrics = formatText(country_lyrics)
    print("Done formatting country data!")
    print("Formatting hip-hop data...")
    hiphop_lyrics = formatText(hiphop_lyrics)
    print("Done formatting hip-hop data!")

    # TESTING
    print("Beginning testing new lyrics...")
    results = calculateTestingProbabilities(testingEntries, country_lyrics, hiphop_lyrics)
    printStatistics(results, testingEntries)
    print("Done testing new lyrics!")

###############################################################################################################
if __name__ == "__main__":
    # execute only if run as a script
    main()