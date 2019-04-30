# Name: Palmer Robins & Sara Camili

import csv

from Parser import readCSVFile
from GenerateFeatureVectors import generateTrainingDicts_LanguageModels, generateTrainingDicts_Bayes
from handleTestLyrics import classifyLyric

def main():

    country_lyrics = []
    hiphop_lyrics = []

    entries = readCSVFile("lyrics.csv")

    for lyric in entries:
        if (lyric[4] == "Country"):
            country_lyrics.append(lyric[5])
        else:
            hiphop_lyrics.append(lyric[5])

    # print("Entries labeled as country: " + str(len(country_lyrics)))
    # print("Entries labeled as hip-hop: " + str(len(hiphop_lyrics)))

    # Generate probability dictionaries for unigrams and bigrams using language models
    LM_dictionaries = generateTrainingDicts_LanguageModels(hiphop_lyrics, country_lyrics)

    # Generate probability dictionaries using a Bayes model
    bayes_dictionaries = generateTrainingDicts_Bayes(hiphop_lyrics, country_lyrics)

    # Assume that the tester file is passed in via the command file - parse it somehow?
    testFile = sys.argv[1]

    # Use these probability dictionaries to classify new lyrics
    for item in testFile:
        classification = classifyLyric(item, LM_dictionaries, bayes_dictionaries)
        print(classification)           #TODO: what do we want to do with the result

###############################################################################################################
if __name__ == "__main__":
    # execute only if run as a script
    main()