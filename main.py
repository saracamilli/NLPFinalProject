# Name: Palmer Robins & Sara Camili

import csv

from Parser import readCSVFile
from GenerateFeatureVectors import generateTrainingFeatureVectors

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

    featureVects = generateTrainingFeatureVectors(country_lyrics, hiphop_lyrics)

    hipHopFeatureVects = featureVects[0]
    countryFeatureVects = featureVects[1]

    # TODO: somehow fill this into the afffilecreation func

    # Assume that the tester file is passed in via the command file - parse it somehow?
    testFile = sys.argv[1]
    #parsedFile =

###############################################################################################################
if __name__ == "__main__":
    # execute only if run as a script
    main()