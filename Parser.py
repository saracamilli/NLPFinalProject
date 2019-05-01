# Name: Palmer Robins & Sara Camili

import csv

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

###############################################################################################################
# @brief Given a filename corresponding to the selected csv file of lyrics, this function attempts to open
# the file and return a list
###############################################################################################################

def readCSVFile(filename):
    print ("Reading in the training csv file...")
    entries = []

    try:
        with open(filename, 'r') as lyricsFile:
            # creating a csv reader object
            csvreader = csv.reader(lyricsFile)

            # extracting field names through first row
            fields = csvreader.next()

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

    print("Done reading the training file!")

    return entries

def formatText(lyrics):
    formattedLyrics = []
    porter = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    for lyric in lyrics:
        if lyrics.index(lyric) > 30000:
            break
        try:
            tokens = word_tokenize(lyric)
            for word in tokens:
                word = lemmatizer.lemmatize(word)
            words = [word for word in tokens if word.isalpha()]
            stemmed = ' '.join([porter.stem(word) for word in tokens])
            formattedLyrics.append(stemmed)
        except UnicodeDecodeError as e:
            continue
    return formattedLyrics
