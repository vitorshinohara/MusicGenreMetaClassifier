from keras.utils import to_categorical
import numpy as np

def generateGenreDict(labelsFile):
	genre_dict = {}
	with open(labelsFile, 'r') as file:
		lines = formatLines(file.readlines())
		previousGenre = ''
		index = 0

		for line in lines:
			genre = line.split('\t')[1]

			if genre != previousGenre and genre not in genre_dict:

				genre_dict[genre] = index
				index = index + 1
				previousGenre = genre

	return genre_dict


def extractFeaturesAndLabelFromFile(fileName, labelsPath):
	""" Get features from ONE FILE """
	genre_dict = generateGenreDict(labelsPath)

	labels = []
	fileSequenceVotes = []
	train_lenght = []

	with open(fileName, 'r') as file:
		lines = formatLines(file.readlines())
		lastMusic = lines[0].split(';')[0]
		musicVotes = []

		for i, line in enumerate(lines):

			music, genre = getMusicAndGenre(line)
			genre = genre_dict[genre]

			if music != lastMusic:
				musicVotes = np.array(musicVotes).reshape(-1,1)
				fileSequenceVotes.append(musicVotes)
				musicVotes = []
				musicVotes.append(genre)

				realGenre = getRealMusicGenre(labelsPath, music)
				realGenre = genre_dict[realGenre]
				labels.append(realGenre)

			else:
				musicVotes.append(genre)

			lastMusic = music

			if i == len(lines) - 1:
				musicVotes = np.array(musicVotes).reshape(-1,1)
				fileSequenceVotes.append(musicVotes)
				realGenre = getRealMusicGenre(labelsPath, music)
				realGenre = genre_dict[realGenre]
				labels.append(realGenre)


	class_number = len(np.unique(labels))

	categoricalLabels = [ to_categorical(label, class_number) for label in labels ]
	return fileSequenceVotes, categoricalLabels

def getRealMusicGenre(filePath, musicPath):

	with open(filePath, 'r') as file:
		lines = formatLines(file.readlines())
		for line in lines:
			if musicPath.split('/')[-1] in line:
				genre = line.split('\t')[1]
				return genre

def formatLines(lines):
	lines_formated = []
	for line in lines:
		lines_formated.append(line.replace('\n', ''))
	return lines_formated

def getMusicAndGenre(line):
	lineSplitted = line.split(';')
	return lineSplitted[0], lineSplitted[1]
