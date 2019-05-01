# Name: Palmer Robins & Sara Camili

import csv
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
#nltk.download('stopwords')

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
            fields = next(csvreader)

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
            # Tokenize the lyric
            tokens = word_tokenize(lyric)
            # Convert to lower case & stem
            tokens = [porter.stem(t) for t in tokens]
            # remove punctuation from each word
            table = str.maketrans('','', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            # remove all tokens that are not alphabetic
            words = [word for word in stripped if word.isalpha()]
            stop_words = set(stopwords.words('english'))
            words = [w for w in tokens if not w in stop_words]
            formattedLyrics.append(words)
        except UnicodeDecodeError as e:
            continue
    return formattedLyrics
