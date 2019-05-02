# Name: Palmer Robins & Sara Camili

# THIS FILE HANDLES UNSEEN LYRICS

from __future__ import division
from helpersForLMAndBayes import nGramCounts, computeProb_LM, computeProb_Bayes
from math import log
from numpy import mean, std
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
import random


#################################################################################################################
# Given new tester entries, this function computes the probability of each given lyric being from a country or
#  ahip hop song. It does this by going computing
#################################################################################################################
def calculateTestingProbabilities(testingEntries, country_lyrics, hiphop_lyrics):
    n = 2

    # Get nGram counts for country training data
    print("Getting nGram and word counts...")
    country_nGramCounts = nGramCounts(country_lyrics, n)
    country_nMinus1GramCounts = nGramCounts(country_lyrics, n - 1)
    # Get nGram counts for hip-hop training data
    hiphop_nGramCounts = nGramCounts(hiphop_lyrics, n)
    hiphop_nMinus1GramCounts = nGramCounts(hiphop_lyrics, n - 1)

    # Get total and estimated word counts using helper function
    wordCounts = getWordCounts(country_lyrics, hiphop_lyrics, country_nGramCounts, country_nMinus1GramCounts, \
                               hiphop_nGramCounts, hiphop_nMinus1GramCounts)
    country_TotalWordCount = wordCounts[0]
    hiphop_TotalWordCount = wordCounts[1]
    country_estimatedUnknownWordCount = wordCounts[2]
    hiphop_estimatedUnknownWordCount = wordCounts[3]
    print("Done getting nGram and word counts!")

    # Calculate average word/lines length means of each corpus - uncomment to include this!
    #meanWordLength_country = calculateAvgWordLength(country_lyrics)
    #meanWordLength_hiphop = calculateAvgWordLength(hiphop_lyrics)
    #meanLineLength_country = calculateAvgLineLength(country_lyrics)
    #meanLineLength_hiphop = calculateAvgLineLength(hiphop_lyrics)

    # Incorporate Naive Bayes features as well
    print("Calculating Naive-Bayes probabilities...")
    countryProbs_Bayes = calculateSongProbability_BAYES(testingEntries, country_lyrics, country_estimatedUnknownWordCount)
    hiphopProbs_Bayes = calculateSongProbability_BAYES(testingEntries, hiphop_lyrics, hiphop_estimatedUnknownWordCount)
    print("Done calculating Naive-Bayes probabilities!")

    results = []    # Stores newly classified test sentences

    counter = 0
    print("Calculating final probabilities using keywords, NB, and LMs...")
    for entry in testingEntries:
        if counter > 30000:
            break

        # Probability that any given sentence is either country or hip-hop (category probability)
        countryProb = -log(len(country_lyrics) / (len(country_lyrics) + len(hiphop_lyrics)))
        hiphopProb = -log(len(hiphop_lyrics) / (len(country_lyrics) + len(hiphop_lyrics)))

        lyric = entry[3:]
        words = lyric.split()

        for i in range(0, len(words) - 2):
            nGram = words[i] + " " + words[i + 1]
            history = words[i].strip()
            countryProb += computeProb_LM(nGram, country_nGramCounts.get(nGram), country_nMinus1GramCounts.get(history),
                                       country_TotalWordCount, country_estimatedUnknownWordCount)
            hiphopProb += computeProb_LM(nGram, hiphop_nGramCounts.get(nGram), hiphop_nMinus1GramCounts.get(history),
                                      hiphop_TotalWordCount, hiphop_estimatedUnknownWordCount)

        # Use both Naive Bayes and LM probabilities together
        countryProb = (countryProb + countryProbs_Bayes[testingEntries.index(entry)]) / 2
        hiphopProb = (hiphopProb + hiphopProbs_Bayes[testingEntries.index(entry)]) / 2

        # Extract and use keyword features; add to probability for every matching keyword in the
        # corresponding genre
        keywordFeat = extractKeywordFeatures(words)
        for i in keywordFeat[0]:
            if i == 1:
                hiphopProb += 0.05
        for i in keywordFeat[1]:
            if i == 1:
                countryProb += 0.05

        # UNCOMMENT THE SECTIONS BELOW to include average word length and average line length features
        # Calculate whether entry is more likely to be country or hip hop on the basis of avg. word length
        #distrWord = calcMoreLikelyWordLengthDistrib(meanWordLength_country, meanWordLength_hiphop, entry)
        #if distrWord == "country":
            #countryProb += 0.05
        #if distrWord == "hiphop":
            #hiphopProb += 0.05

        # Calculate whether entry is more likely to be country or hip hop on the basis of avg. line length
        #distrLine = calcMoreLikelyLineLengthDistrib(meanLineLength_country, meanLineLength_hiphop, entry)
        #if distrLine == "country":
            #countryProb += 0.05
        #if distrLine == "hiphop":
            #hiphopProb += 0.05

        # Incorporate Naive Bayes
        countryProb = (countryProb + countryProbs_Bayes[testingEntries.index(entry)]) / 2
        hiphopProb = (hiphopProb + hiphopProbs_Bayes[testingEntries.index(entry)]) / 2

        # Compare probabilities
        if (countryProb > hiphopProb):
            results.append("c: " + lyric)
        elif (hiphopProb > countryProb):
            results.append("h: " + lyric)
        else:
            if (random.random() > 0.5):
                results.append("c: " + lyric)
            else:
                results.append("h: " + lyric)
        counter += 1
    print("Done calculating final probabilities!")

    return results

#################################################################################################################
# Get the total word counts for both classes, as well as an estimation of unknown word counts
#################################################################################################################
def getWordCounts(country_lyrics, hiphop_lyrics, country_nGramCounts, country_nMinus1GramCounts, \
    hiphop_nGramCounts, hiphop_nMinus1GramCounts):

    country_TotalWordCount = 0
    hiphop_TotalWordCount = 0
    country_estimatedUnknownWordCount = 0
    hiphop_estimatedUnknownWordCount = 0

    for count in country_nMinus1GramCounts.values():
        if count <= 5:
            country_estimatedUnknownWordCount += 1
        else:
            country_TotalWordCount += count
    for count in hiphop_nMinus1GramCounts.values():
        if count <= 5:
            hiphop_estimatedUnknownWordCount += 1
        hiphop_TotalWordCount += count

    return(country_TotalWordCount, hiphop_TotalWordCount, country_estimatedUnknownWordCount, \
        hiphop_estimatedUnknownWordCount)

#################################################################################################################
# Given counts of all the unigrams, along with the overall size of the vocabulary in the text,
# generate probabilities using Naive Bayes with unigram features..
#################################################################################################################
def calculateSongProbability_BAYES(testingEntries, lyrics, unkWordCount):

    # Get unigram counts
    unigramCounts = nGramCounts(lyrics, 1)

    probabilities = []

    # Loop through all lyrical entries
    for entry in testingEntries:
        prob = 0
        # Loop through all the words in the sentence
        for word in entry.split():
            # Compute probabilities using helper function
            prob += computeProb_Bayes(word, unigramCounts, unkWordCount)
        probabilities.append(prob)

    return probabilities

#################################################################################################################
# Given a text block, extracts presence of keywords for both hip hop and country features and creates two
# feature vectors which it returns as (hip hop vector, country vector)
#################################################################################################################
def extractKeywordFeatures(textBlock):

    # Extracted from the Internet - need more hip hop keywords?
    hipHopKeywords = ["chopper", "stunting", "flexing", "mane", "trill", "trapping", "balling" \
        "realest", "homie", "snitch", "biggie", "chains", "grind", "nigga", "shit", "bitch", "skrrt", \
            "never", "fuck", "hit", "money", "ass", "big", "real", "motherfucker", "hustle", \
                "cigarette", "war", "bullet", "slap", "dick", "suicide", "murder", "shawty", "New York", \
                    "Los Angeles", "Harlem", "Compton"]

    countryKeywords = ["country", "drinkin", "cowboy", "tailgates" "tobacco", \
         "memory", "drunk", "hotties", "highway", "beer", "whiskey" \
            "little", "away", "dirt", "mud", "chew", "whoa", "southern", "south", "chevy" \
                "redneck", "springsteen", "cruise", "truck", "headlights", "town", "ford" \
                    "radio", "rodeo", "hey", "rolling", "song", "round", "til", "lane", "wind", "backwoods", "boondocks", \
                        "summer", "Georgia", "Alabama", "Carolina", "Tennessee", "Kentucky", "Shenandoah"]

    # Define and initialize feature vectors with 0's
    hipHopFeatureVect = []
    for i in range(len(hipHopKeywords)): hipHopFeatureVect.append(0)
    countryFeatureVect = []
    for i in range(len(countryKeywords)): countryFeatureVect.append(0)

    # Update the feature vectors
    for word in textBlock:
        # Check if the current word is in hip hop keywords list
        if (word in hipHopKeywords):
            # If it is, replace its 0 in the vector with a 1
            index = hipHopKeywords.index(word)
            hipHopFeatureVect[index] = 1

        if (filter(word) in hipHopKeywords):
            # If it is, replace its 0 in the vector with a 1
            index = hipHopKeywords.index(filter(word))
            hipHopFeatureVect[index] = 1

        # Check if the current word is in country keywords list
        if word in countryKeywords:
            # If it is, replace its 0 in the vector with a 1
            index = countryKeywords.index(word)
            countryFeatureVect[index] = 1

        if (filter(word) in countryKeywords):
            # If it is, replace its 0 in the vector with a 1
            index = countryKeywords.index(filter(word))
            countryFeatureVect[index] = 1

    return(hipHopFeatureVect, countryFeatureVect)


#################################################################################################################
# Given average country and hip hop mean word lengths along with an unknown lyrical entry, calculates whether
# the unknown entry is more similar to country or to hip hop on the basis of word length
#################################################################################################################
def calcMoreLikelyWordLengthDistrib(meanWordLength_country, meanWordLength_hiphop, entry):

    # Loop through words in the entry and calculate the mean length
    wordLengths_tester = []
    for word in entry.split():
        wordLengths_tester.append(len(word))
    meanWordLength_tester = mean(wordLengths_tester)

    # Calculate the difference in mean word length between the average for each category
    diffFromCountry = abs(meanWordLength_country - meanWordLength_tester)
    diffFromHiphop = abs(meanWordLength_hiphop - meanWordLength_tester)

    # Compare the differences to each category and return category with smallest distance
    if (diffFromCountry > diffFromHiphop):
        if (diffFromHiphop < 0.5):
            return("hiphop")
    elif (diffFromCountry < diffFromHiphop):
        if (diffFromCountry < 0.5):
            return("country")
    else:
        return("same")

#################################################################################################################
# Give a set of lyrics (either hip hop or country), calculates the average word length (which is returned). We
# will assume that word length is normally distributed around this mean. This will be used to determine which
# distribution a new phrase is most likely to have drawn from.
#################################################################################################################
def calculateAvgWordLength(lyrics):

    words = []          # a list of all the word lengths in the set of lyrics

    for sentence in lyrics:
        for word in sentence:
            words.append(len(word))

    avgWordLength = mean(words)

    return(avgWordLength)

#################################################################################################################
# Given country and hip hop mean line lengths along with an unknown lyrical entry, calculates whether the unknown
# entry is more similar to country or to hip hop on the basis of line length
#################################################################################################################
def calcMoreLikelyLineLengthDistrib(meanLineLength_country, meanLineLength_hiphop, entry):

    # Calculate length of entry
    lineLengths_tester = len(entry)
    meanLineLength_tester = mean(lineLengths_tester)

    # Calculate the difference in mean line length between the average for each category
    diffFromCountry = abs(meanLineLength_country - meanLineLength_tester)
    diffFromHiphop = abs(meanLineLength_hiphop - meanLineLength_tester)

    # Compare the differences to each category and return category with smallest distance
    if (diffFromCountry > diffFromHiphop):
        if (diffFromHiphop < 0.3):
            return("hiphop")
    elif (diffFromCountry < diffFromHiphop):
        if (diffFromCountry < 0.3):
            return("country")
    else:
        return("same")

#################################################################################################################
# Give a set of lyrics (either hip hop or country), calculates the average line length (which is returned). We
# will assume that line length is normally distributed around this mean. This will be used to determine which
# distribution a new phrase is most likely to have drawn from.
#################################################################################################################
def calculateAvgLineLength(lyrics):

    lines = []          # a list of all the line lengths in the set of lyrics

    for sentence in lyrics:
        lines.append(len(sentence))

    avgLineLength = mean(lines)

    return(avgLineLength)

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

	if (inputWord not in stopWords):						# if not a stop word
		psw = porterStemmer.stem(inputWord)				# stem the word
		filteredWord = lemmatizer.lemmatize(psw)	# append the lemmatized and stemmed word

	return (filteredWord)