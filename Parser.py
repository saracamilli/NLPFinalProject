# Name: Palmer Robins & Sara Camili

import csv
import string

import nltk
from nltk.tokenize import word_tokenize


###############################################################################################################
# @brief Given a filename corresponding to the selected csv file of lyrics, this function attempts to open
# the file and return a list
###############################################################################################################

def readCSVFile(filename):

    entries = []
    try:
        with open(filename, 'r') as lyricsFile:
            # creating a csv reader object
            csvreader = csv.reader(lyricsFile)
            next(csvreader)

            # extracting each data row containing country/hip-hop lyrics, one by one
            for entry in csvreader:
                # if the genre field is "country" or "hip-hop"
                if entry[4] == "Country" or entry[4] == "Hip-Hop":
                    sentences = entry[5].splitlines()
                    for line in sentences:
                        if len(line) < 35 or len(line) > 300:
                            continue
                        else:
                            if (entry[4] == "Country"):
                                entries.append("c: " + line)
                            else:
                                entries.append("h: " + line)

    except IOError:
        print("Error: Cannot open the corpus containing the training data.")
        print("Filename requested: " + filename)
        exit(1)

    return entries

def formatText(lyrics):
    formattedLyrics = []
    for lyric in lyrics:
        if lyrics.index(lyric) > 30000:
            break
        try:
            # Tokenize the lyric
            tokens = word_tokenize(lyric)
            # Remove punctuation from each word
            table = str.maketrans('','', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            #remove all tokens that are not alphabetic
            words = [word for word in stripped if word.isalpha()]
            formattedLyrics.append(words)
        except UnicodeDecodeError as e:
            continue
    return formattedLyrics
