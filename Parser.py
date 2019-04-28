# Name: Palmer Robins & Sara Camili

import csv
import operator
import sys

from GenerateFeatureVectors import generateTrainingFeatureVectors

def main():
    entries = readCSVFile("lyrics.csv")
    featureVects = generateTrainingFeatureVectors(entries)

    hipHopFeatureVects = featureVects[0]
    countryFeatureVects = featureVects[1]

    # TODO: somehow fill this into the afffilecreation func

    # Assume that the tester file is passed in via the command file - parse it somehow?
    testFile = sys.argv[1]
    #parsedFile =


###############################################################################################################
# @brief Given a filename corresponding to the selected csv file of lyrics, this function attempts to open
# the file and return a list
###############################################################################################################
def readCSVFile(filename):

    fields = []
    entries = []
    # use a dictionary pairing its classification "country" or "hip-hop" to its value (lyrics)

    try:
        with open(filename, 'r') as lyricsFile:
            # creating a csv reader object
            csvreader = csv.reader(lyricsFile)

            # extracting field names through first row
            fields = csvreader.next()

            # extracting each data row containing country/hip-hop lyrics, one by one
            for entry in csvreader:
                # if the genre field is "country" or "hip-hop"
                if ((entry[4] == "Country") or (entry[4] == "Hip-Hop")):
                    print("Genre: " + str(entry[4]))
                    entries.append(entry)


        # printing the field names
        print('Field names are: ' + ','.join(field for field in fields))

        # printing first 5 rows
        print('\nFirst 5 entries are:\n')
        for entry in entries[:5]:
            # parsing each column of a row
            for col in entry:
                print("%10s"%col),
            print('\n')
    except IOError:
        print("Error: Cannot open the corpus containing the training data.")
        print("Filename requested: " + filename)
        exit(1)

    return entries

###############################################################################################################
# Parses the input file with lyrics given by the user who wants to classify them by genre (must be in the 
# specific format giving in the README)
###############################################################################################################
def parseTesterFile(testFile):
    try:
        f = open(testFile)
        parsedFile = []

        #TODO: FILL THIS IN

        f.close()
        return(parsedFile)

    except IOError:
        print("Error: Cannot open the file containing the tester data.")
        print("Filename requested: " + testFile)
        exit(1)


###############################################################################################################
if __name__ == "__main__":
    # execute only if run as a script
    main()