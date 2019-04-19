# Name: Palmer Robins & Sara Camili

import csv
import operator
import sys

def main():
    readCSVFile("lyrics.csv")

###############################################################################################################
# @brief Given a filename corresponding to the selected csv file of lyrics, this function attempts to open
# the file and return a list
###############################################################################################################
def readCSVFile(filename):

    fields = []
    entries = []

    try:
        with open(filename, 'r') as lyricsFile:
            # creating a csv reader object
            csvreader = csv.reader(lyricsFile)

            # extracting field names through first row
            fields = csvreader.next()

            # extracting each data row one by one
            for entry in csvreader:
                entries.append(entry)

            # get total number of rows
            print("Total no. of rows: %d"%(csvreader.line_num))

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

###############################################################################################################
if __name__ == "__main__":
    # execute only if run as a script
    main()