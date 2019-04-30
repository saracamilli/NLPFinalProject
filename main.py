# Name: Palmer Robins & Sara Camili

import csv

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from Parser import readCSVFile

def main():

    country_lyrics = []
    hiphop_lyrics = []

    entries = readCSVFile("lyrics.csv")

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

    print("Formatting country lyrics...")
    country_lyrics = formatText(country_lyrics)
    print("Done formatting country lyrics!\nFormatting hip-hop lyrics...")

    print("===========  COUNTRY LYRICS ============\n\n")
    for x in range(0,20):
        print(country_lyrics[x] + "\n\n")


    hiphop_lyrics = formatText(hiphop_lyrics)
    print("Done formatting hip-hop lyrics!")

    print("===========  HIP-HOP LYRICS ============\n\n")
    for x in range(0,20):
        print(hiphop_lyrics[x] + "\n\n")

    # Step 1: Format text for LM Processing.
    # Step 2: Generate N-Gram counts
    # Step 3: Train LM using Katz Backoff, with absolute discounting using country and hip-hop lyrics
    # Step 4: Use language model to classify all testing lyrics

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