# Name: Palmer Robins & Sara Camili

# USE FEATURE VECTORS TO FORMULATE AN ARFF FILE
# THIS FILE WILL BE FED INTO WEKA CLASSIFIERS

# Use the feature vectors generated to create an Arf file that will be fed into
# WEKA to create a classifier
def createArffFile(featureVectors, senseDict, trainOrTest):

	# Create new arf file!
	if (trainOrTest == "train"):
		file = open("trainFile.arff", "w")		# the "w" is for new file creation if name doesn't already exist
	elif (trainOrTest == "test"):
		file = open("testFile.arff", "w")
	else:
		print("error in train or test file distinguishment")

	file.write("@RELATION lyrics\n")			# add an @RELATION

	# Add attributes to the file
	file.write("@ATTRIBUTE collocationalWOFeatures STRING\n")
	file.write("@ATTRIBUTE collocationalWPOSFeatures STRING\n")
	file.write("@ATTRIBUTE cooccurrenceWOFeatures STRING\n")
	file.write("@ATTRIBUTE cooccurrenceWPOSFeatures STRING\n")
	file.write("@ATTRIBUTE collocAndCooccWOFeatures STRING\n")
	file.write("@ATTRIBUTE collocAndCooccWPOSFeatures STRING\n")

	# Add the possible classes
	senses = set(senseDict.values())
	stringSenses = ""
	for s in senses:
		stringSenses = stringSenses + "," + str(s)
	file.write("@ATTRIBUTE class {" + stringSenses + "}\n")

	# Add the data!
	file.write("@DATA\n")

	counter = 1
	# Loop through each set of 6 vectors in the full set of vectors
	for setOfVect in featureVectors:

		sense = list(senseDict.items())[counter]
		senseTrue = sense[1]

		# Loop through each of these 6 vectors and convert to a string
		strVect = ""
		for indivVect in setOfVect:
			#print("indivVect: " + str(indivVect))
			littleStrVect = "'" + ",".join(str(v) for v in indivVect) + "'"		# convert to comma-delin. string
			strVect = strVect + ", " + littleStrVect

		strVect = strVect + "," + senseTrue

		file.write(strVect + "\n")
