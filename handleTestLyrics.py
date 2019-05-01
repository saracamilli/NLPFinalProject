# Name: Palmer Robins & Sara Camili
from math import log
from GenerateFeatureVectors import LanguageModel, formatText

# THIS FILE HANDLES THE UNSEEN LYRICS. There is a command center function that uses a language model, Bayes
# probability, and keyword feature extraction to create an as-accurate-as-possible classification.

#################################################################################################################
# Compile all the features we are considering in order to make an ultimate judgement on whether the unseen
# sentence is country or hip hop. Use the probabilities from the language models, the Bayes model, and the
# keyword features.
# Input: a lyrical sentence of unknown genre
# Output: a classification string of "hip hop" or "country"
#################################################################################################################
class Tester:

    def __init__(self, sentences, numCountry, numHipHop):
        self.sentences = formatText(sentences)
        self.results = []
        self.numCountry = numCountry
        self.numHipHop = numHipHop


    def classifyTestLyric():
        counter = 0
        for entry in sentences:
            if counter > 30000:
                break
            # Probability that any given sentence is either country or hip-hop
            countryProb = -log(numCountry / (numCountry + numHipHop))
            hiphopProb = -log(numHipHop / (numHipHop + numCountry))
            lyric = entry[3:]
            words = lyric.split()

            for i in range(0, len(words) - 2):
                nGram = words[i] + " " + words[i + 1]
                history = words[i].strip()
                countryProb += computeProb(nGram, country_nGramCounts.get(nGram), country_nMinus1GramCounts.get(history),
                                        country_TotalWordCount, country_estimatedUnknownWordCount)
                hiphopProb += computeProb(nGram, hiphop_nGramCounts.get(nGram), hiphop_nMinus1GramCounts.get(history),
                                        hiphop_TotalWordCount, hiphop_estimatedUnknownWordCount)
            if (countryProb > hiphopProb):
                results.append("c: " + lyric)
            elif (hiphopProb > countryProb):
                results.append("h: " + lyric)
            else:
                print("The probabilities are the same")
                if (random.choice(1,-1) > 0):
                    results.append("c: " + lyric)
                else:
                    results.append("h: " + lyric)
            counter += 1


    # NOW, SOMEHOW COMBINE ALL THESE PROBABILTIES IN A MEANINGFUL WAY
    #################################################################################################################
    # Given counts of all the bigrams and unigrams, along with the overall size of the vocabulary in the text,
    # generate probabilities using Katz-backoff with absolute discounting
    #################################################################################################################
    def computeProb(entry, nGramCount, nMinus1GramCount, vocabSize, unknownWordCount):
        words = entry.split()   # Split the n-gram into words
        history = words[0]

        if nGramCount is None or nMinus1GramCount is None:
            if (nMinus1GramCount is None):
                prob = -log(unknownWordCount / vocabSize)
                return prob
            else:
                prob = -log(nMinus1GramCount / vocabSize)
                return prob

        try:
            return -log((nGramCount - 0.75) / nMinus1GramCount)
        except TypeError:
            print(entry)
            print(nGramCount)
            print(history)
            print (nMinus1GramCount)

#################################################################################################################
# Given a new tester text, this helper function takes a dictionary of the unigram, bigram, or trigram counts of
# the new text and loops through all the entries. It takes the product of their probabilities,
# and then multiplies them together. It then returns this final probability of the sentence being hip hop or
# country.
#################################################################################################################
def calculateSongProbability_BAYES(song, probDict):

    product = 1

    # Loop through all the words in the sentence
    for word in song.split():

        # Access probabilities of each word in that sentence
        prob = probDict.get(word)

        # If prob is None, change to very small probability
        if (prob is None):
            prob = 0.00000001

        product = product * prob

    return(product)

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
        "road", "memory", "drunk", "got", "know", "highway", "cold", "beer" \
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