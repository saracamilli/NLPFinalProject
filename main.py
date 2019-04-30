# Name: Palmer Robins & Sara Camili
from __future__ import division
import csv

from math import log

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from Parser import readCSVFile
from GenerateFeatureVectors import computeProb, nGramCounts
from handleTestLyrics import classifyLyric

def main():

    country_lyrics = []
    hiphop_lyrics = []
    n = 2

    # Store country/hip-hop lyrics as sentences in list
    entries = readCSVFile("lyrics.csv")

    # use half the list for training, half for testing
    trainingEntries = entries[:len(entries)//2]
    testingEntries = entries[len(entries)//2:]

    # Use the genre label to insert each lyric into the country or hip-hop dataset
    print("Classifying training data...")
    for lyric in testingEntries:
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

    #print("===========  COUNTRY LYRICS 0-20 ============\n\n")
    #for x in range(0,20):
       # print(country_lyrics[x] + "\n\n")


    hiphop_lyrics = formatText(hiphop_lyrics)
    print("Done formatting hip-hop lyrics!")

    #print("===========  HIP-HOP LYRICS 0-20 ============\n\n")
    #for x in range(0,20):
       # print(hiphop_lyrics[x] + "\n\n")

    # Step 2: Generate N-Gram counts
    # Step 3: Train LM using Katz Backoff, with absolute discounting using country and hip-hop lyrics
    # Step 4: Use language model to classify all testing lyrics

    # Generate probability dictionaries for unigrams and bigrams using language models
    # LM_dictionaries = generateTrainingDicts_LanguageModels(hiphop_lyrics, country_lyrics)

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
    country = 0
    hip_hop = 0

    print("Number of Testing Entries: " + str(len(testingEntries)))
    for entry in testingEntries:

        # Probability that any given sentence is either country or hip-hop
        countryProb = log(len(country_lyrics) / (len(country_lyrics) + len(hiphop_lyrics)))
        hiphopProb = log(len(hiphop_lyrics) / (len(country_lyrics) + len(hiphop_lyrics)))

        entry = entry[3:]
        words = entry.split()

        for i in range(0, len(words) - 2):
            entry = words[i] + " " + words[i + 1]
            history = words[i].strip()
            countryProb += computeProb(entry, country_nGramCounts.get(entry), country_nMinus1GramCounts.get(history),
                                       country_TotalWordCount, country_estimatedUnknownWordCount)
            hiphopProb += computeProb(entry, hiphop_nGramCounts.get(entry), hiphop_nMinus1GramCounts.get(history),
                                      hiphop_TotalWordCount, hiphop_estimatedUnknownWordCount)
            print("Country Prob: " + str(countryProb))
            print("Hip-Hop Prob: " + str(hiphopProb))



    # Generate probability dictionaries using a Bayes model
    # bayes_dictionaries = generateTrainingDicts_Bayes(hiphop_lyrics, country_lyrics)

    # Use these probability dictionaries to classify new lyrics
    #for entry in testingEntries:
        # Remove genre tag before using as a testing sentence
        #entry = entry[3:]
        #classification = classifyLyric(entry, LM_dictionaries, bayes_dictionaries)
        #print("Lyric:  " + entry)
        #print("Classification:  " + str(classification) + "\n")           # TODO: what do we want to do with the result

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