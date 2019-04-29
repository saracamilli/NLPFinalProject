# Name: Palmer Robins & Sara Camili

import nltk

import codecs
from nltk.tokenize import PunktSentenceTokenizer,sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download('wordnet')

#################################################################################################################
# Generate feature vectors using n-gram counts and keywords (using helpers), then returns these feature vectors
# as a list. These will be used to create the arff file to feed to weka's machine learning classifier
#################################################################################################################
def generateTrainingFeatureVectors(countryEntries, hiphopEntries):

    # Lists of all the training feature vectors for country and hip hop (consists of both n-gram features
    # and keyword features)
    # TODO: is this the way to go? I kinda forget how this is supposed to work, please help

    # Loop through all entries
    for entry in entries:

        # Get n-gram features
        ngramVects = getNGramFeatures(entry)

        # Get keyword features
        keywordVects = extractKeywordFeatures(entry)

        # If this is a hip hop lyrical block, append the vectors to this list
        if (entry[4] == "Hip Hop"):
            allHipHopVectors.append(ngramVects).append(keywordVects)
        # Otherwise, this is a country lyrical block
        else:
            allCountryVectors.append(ngramVects).append(keywordVects)

    return(allHipHopVectors, allCountryVectors)


#################################################################################################################
# Computes n-gram feature vectors for unigrams, bigrams, and trigrams in the training lyrical text
# Input: a block of training lyrical text from either country or hip hop
# Output: unigram, bigram, and trigram feature vectors
# TODO: do we want to use the n-gram counts somehow?
#################################################################################################################
def getNGramFeatures(textBlock):

    # Generate lists of all the unigrams, bigrams, and trigrams in the given training lyrical block
    unigrams = list(ngramCounts(textBlock, 1).keys())
    bigrams = list(ngramCounts(textBlock, 2).keys())
    trigrams = list(ngramCounts(textBlock, 3).keys())

    return(unigrams, bigrams, trigrams)

#################################################################################################################
# Computes n-gram counts for any n for any given corpus
# Input: integer n, corpus text filename
# Output: count of n-word sequences
#################################################################################################################
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

#################################################################################################################
# Given a text block, extracts presence of keywords for both hip hop and country features and creates two
# feature vectors which it returns as (hip hop vector, country vector)
# TODO: should we change this to frequency, or is presence good enough?
#################################################################################################################
def extractKeywordFeatures(textBlock):

    # Extracted from the Internet - need more hip hop keywords?
    hipHopKeywords = ["chopper", "stunting", "flexing", "mane", "trill", "trapping", "balling" \
        "realest", "homie", "snitch", "biggie", "grind", "nigga", "shit", "bitch", "skrrt", \
            "never", "fuck", "hit", "money", "ass", "big", "real"]

    countryKeywords = ["ride", "baby", "oh", "tobacco", "windows", "blown", \
        "road", "memory", "windows", "drunk", "got", "know", "highway", "cold", "beer" \
            "little", "away", "dirt", "town", "chew", "whoa", "plane", "southern", "south" \
                "redneck", "springsteen", "cruise", "truck", "headlights", "town" \
                    "radio", "hey", "rolling", "song", "round", "til", "lane", "wind"]

    # Define and initialize feature vectors with 0's
    hipHopFeatureVect = []
    for i in range(len(hipHopKeywords)): hipHopFeatureVect.append(0)
    countryFeatureVect = []
    for i in range(len(countryKeywords)): countryFeatureVect.append(0)

    # Update the feature vectors
        # NOTE: the words are not filtered because, esp. in the case of hip hop, the
        # suffixes can be meaningful (i.e. balling, not ball)
    for word in textBlock:
        # Check if the current word is in hip hop keywords list
        if word in hipHopKeywords:
            # If it is, replace its 0 in the vector with a 1
            index = hipHopKeywords.index(word)
            hipHopFeatureVect[index] = 1

        # Check if the current word is in country keywords list
        if word in countryKeywords:
            # If it is, replace its 0 in the vector with a 1
            index = countryKeywords.index(word)
            countryFeatureVect[index] = 1

    return(hipHopFeatureVect, countryFeatureVect)

#################################################################################################################
# Given an input word, returns a token that is "stemmed" & "lemmatized" (contains only
# the morphologically correct root word of the original word)
#################################################################################################################
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