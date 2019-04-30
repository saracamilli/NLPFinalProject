# Name: Palmer Robins & Sara Camili

import csv

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from Parser import readCSVFile
from GenerateFeatureVectors import generateTrainingDicts_LanguageModels, generateTrainingDicts_Bayes
from handleTestLyrics import classifyLyric

def main():

    country_lyrics = []
    hiphop_lyrics = []

    # Store country/hip-hop lyrics as sentences in list
    entries = readCSVFile("lyrics.csv")

    # use half the list for training, half for testing
    trainingEntries = entries[:len(entries)//2]
    testingEntries = entries[len(entries)//2:]

    # Use the genre label to insert each lyric into the country or hip-hop dataset
    print("Classifying training data...")
    for lyric in entries:
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

    print("===========  COUNTRY LYRICS 0-20 ============\n\n")
    for x in range(0,20):
        print(country_lyrics[x] + "\n\n")


    hiphop_lyrics = formatText(hiphop_lyrics)
    print("Done formatting hip-hop lyrics!")

    print("===========  HIP-HOP LYRICS 0-20 ============\n\n")
    for x in range(0,20):
        print(hiphop_lyrics[x] + "\n\n")

    # Step 2: Generate N-Gram counts
    # Step 3: Train LM using Katz Backoff, with absolute discounting using country and hip-hop lyrics
    # Step 4: Use language model to classify all testing lyrics

    # Generate probability dictionaries for unigrams and bigrams using language models
    LM_dictionaries = generateTrainingDicts_LanguageModels(hiphop_lyrics, country_lyrics)

    # Generate probability dictionaries using a Bayes model
    bayes_dictionaries = generateTrainingDicts_Bayes(hiphop_lyrics, country_lyrics)

    # Use these probability dictionaries to classify new lyrics
    for entry in testingEntries:
        classification = classifyLyric(item, LM_dictionaries, bayes_dictionaries)
        print(classification)           # TODO: what do we want to do with the result

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