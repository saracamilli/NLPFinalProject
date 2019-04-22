# Name: Palmer Robins & Sara Camili

import nltk

import codecs
from nltk.tokenize import PunktSentenceTokenizer,sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Given an input word, returns a token that is "stemmed" & "lemmatized" (contains only
# the morphologically correct root word of the original word)
def filter(inputWord):

	filteredWord = ""		# Will hold the new root word from orig. word

	lemmatizer = WordNetLemmatizer()
	porterStemmer = PorterStemmer()

	stopWords = set(stopwords.words("english"))		# set stop words (high frequency words that don't contribute
													# to meaning of sentence, such as 'a', 'the', 'an', etc.)

	if (inputWord not in stopWords):					# if not a stop word
		psw = porterStemmer.stem(inputWord)				# stem the word
		filteredWord = lemmatizer.lemmatize(psw)	# append the lemmatized and stemmed word

	return (filteredWord)


# Computes n-gram counts for any n for any given corpus
# Input: integer n, corpus text filename
# Output: count of n-word sequences
def ngramCounts(textBlock, n):
    
    myCountDict = {}                        # will hold all the n-grams and their counts
                                                    
    # Loop through and find all the possible n-grams, adding them
    # to its dictionary. If it already exists in the dictionary, update
    # its count
    counter = 0
    for word in textBlock:             # loop through all sentences in file
        currNGram = []                      # will be used to hold the current n-gram
                                                # (under construction)
        currNGramStr = ""                   # will hold the converted string version of the n-gram

        currNGram.append(filter(word))      # filter the word and add it to current n-gram

        # If the end of the file has not yet been reached...
        if (counter + n <= len(textBlock)):                 
            endIndex = counter + n
            for theNext in range(counter+1, endIndex):        # loop through next n-1 subsequent words
                currNGram.append(textBlock[theNext])          # put all these words together into an n-gram
            currNGramStr = ' '.join(str(w) for w in currNGram)          # convert to a string
            currNGramStr.replace("[","").replace("]","").replace("'","")

            flag = 0                            # flag for whether we've seen this n-gram before
            for key in myCountDict:              # loop through all n-grams in dictionary                     
                if (key == currNGramStr):                   # if we've encountered this n-gram before...
                    currCount = myCountDict[currNGramStr]      # find it's current count
                    myCountDict.update({currNGramStr : currCount + 1})   # update the dictionary with count + 1
                    flag = 1
                    break

            if flag == 0:
                myCountDict.update({currNGramStr : 1})        # otherwise, add new entry with count 1             

            counter = counter + 1                   

    # Return this dictionary with all n-grams and their counts
    return myCountDict

# Extract counts of keyword features
def extractKeywordFeatures(textBlock):

    # Get these from the Internet?
    hipHopKeywords =[]
    countryKeywords = []