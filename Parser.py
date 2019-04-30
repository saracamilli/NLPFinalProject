# Name: Palmer Robins & Sara Camili

import csv

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
                        if len(line) < 20 or len(line) > 300:
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
