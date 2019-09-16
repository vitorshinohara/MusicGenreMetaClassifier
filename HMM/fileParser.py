import numpy as np

def getMusicAndGenre(line):
    """ Get music and genre from a formated line """
    lineSplitted = line.split(';')
    return lineSplitted[0], lineSplitted[1]


def formatLines(lines):
    """ Remove \n at end of lines """
    return [line.replace('\n', '') for line in lines]


def generateGenreDict(labelsFile):
    """ Generate python's dictionary to convert genre into a integer value """
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

def getRealMusicGenre(filePath, musicPath):
    """ Search the labels file to get the real music genre """
    with open(filePath, 'r') as file:
        lines = formatLines(file.readlines())
        for line in lines:
            if musicPath.split('/')[-1] in line:
                genre = line.split('\t')[1]
                break
    if genre == None:
        print("Erro")
    return genre


def extractFeaturesAndLabelFromFile(fileName, labelsPath, genre_dict):
    """ Get features from ONE FILE
        Params:
            - fileName: file to feature extraction
            - labelsPath: file to check the real genre
            - genre_dict: dict mapping a genre to a integer

        Methods return train_X, train_Y and train_length from one file
    """

    labels, fileSequenceVotes, train_lenght = [], [], []

    sequence_number = fileName.split('/')[2].split('_')[0].split('-')[0]
    sequence_number = int(sequence_number)

    with open(fileName, 'r') as file:

        lines = formatLines(file.readlines())
        lastMusic = lines[0].split(';')[0]
        musicVotes = []
        for i, line in enumerate(lines):

            music, genre = getMusicAndGenre(line)
            genre = genre_dict[genre]

            if music != lastMusic:
                musicVotes = np.array(musicVotes).reshape(-1, 1)
                fileSequenceVotes.append(musicVotes)
                musicVotes = []
                musicVotes.append(genre)

                realGenre = getRealMusicGenre(labelsPath, music)
                realGenre = genre_dict[realGenre]
                labels.append(realGenre)

                train_lenght.append(sequence_number)

            else:
                musicVotes.append(genre)

            lastMusic = music

            if i == len(lines) - 1:
                musicVotes = np.array(musicVotes).reshape(-1, 1)
                fileSequenceVotes.append(musicVotes)
                musicVotes = []
                musicVotes.append(genre)

                realGenre = getRealMusicGenre(labelsPath, music)
                realGenre = genre_dict[realGenre]
                labels.append(realGenre)

                train_lenght.append(sequence_number)

    return fileSequenceVotes, labels, train_lenght

