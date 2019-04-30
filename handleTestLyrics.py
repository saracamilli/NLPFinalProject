# Name: Palmer Robins & Sara Camili

# THIS FILE HANDLES THE UNSEEN LYRICS. There is a command center function that uses a language model, Bayes 
# probability, and keyword feature extraction to create an as-accurate-as-possible classification.

#################################################################################################################
# Compile all the features we are considering in order to make an ultimate judgement on whether the unseen 
# sentence is country or hip hop. Use the probabilities from the language models, the Bayes model, and the
# keyword features.
# Input: a lyrical sentence of unknown genre
# Output: a classification string of "hip hop" or "country"
#################################################################################################################
def classifyLyric(sentence, LM_dictionaries, bayes_dictionaries):

    # Get keyword features
    keywordVects = extractKeywordFeatures(sentence)

    # Get language model features
    bigram_hipHop_dict = LM_dictionaries[0]
    unigram_hipHop_dict = LM_dictionaries[1]
    bigram_country_dict = LM_dictionaries[2]
    unigram_country_dict = LM_dictionaries[3]

    bigramProbs = calculateSongProbability_LANG_MODEL(sentence, bigram_hipHop_dict, bigram_country_dict, "bigram")
    unigramProbs = calculateSongProbability_LANG_MODEL(sentence, unigram_hipHop_dict, unigram_country_dict, "unigram")

    # Get Bayes features
    bayes_hipHop_dict = bayes_dictionaries[0]
    bayes_country_dict = bayes_dictionaries[1]

    bayes_hipHop_Prob = calculateSongProbability_BAYES(sentence, bayes_hipHop_dict)
    bayes_country_Prob = calculateSongProbability_BAYES(sentence, bayes_country_dict)

    # NOW, SOMEHOW COMBINE ALL THESE PROBABILTIES IN A MEANINGFUL WAY


#################################################################################################################
# Given a new tester sentence, this helper function uses the probability dictionaries to compute the probability
# of a given sentence being from a country and hip hop song. Does this by going through each bi/unigram in the 
# line and computing a probability of that bi/unigram being in each category using the Katz-Backoff probabilities
# in the dictionary
#################################################################################################################
def calculateSongProbability_LANG_MODEL(sentence, hipHopProbDict, countryProbDict, bigramOrUnigram):

    currentGram = ""                  # will hold the current bigram

    totalProb_hipHop = 1
    totalProb_country = 1

    # Loop through all bigrams in the line and retrieve their probability in each dictionary
    counter = 0
    for word in sentence.split():

        # This is the bigram version
        if (bigramOrUnigram == "bigram"):
            # If we still have remaining words...
            try:
                # Form bigram
                nextWord = sentence.split()[counter+1]
                currentGram = word + " " + nextWord

                # Check for probabilities in dictionaries 
                prob_hipHop = hipHopProbDict.get(currentGram)
                prob_country = countryProbDict.get(currentGram)

                pass

             # There's not enough words left!
            except Exception as e:
                break

        # This is the unigram version
        else:
            currentGram = word

            # Check for probabilities in dictionaries 
            prob_hipHop = hipHopProbDict.get(currentGram)
            prob_country = countryProbDict.get(currentGram)

        # If there is no match from the training text, set the probability of the bigram to 
        # a very small probability
        if (prob_hipHop is None):
            prob_hipHop = 0.00000001
        if (prob_country is None):
            prob_country = 0.00000001
        
        # Compute the total probability with this gram included
        totalProb_hipHop = totalProb_hipHop * prob_hipHop
        totalProb_country = totalProb_country * prob_country
       
    return(totalProb_hipHop, totalProb_country)

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