# coding: utf8

from hmmlearn.hmm import GaussianHMM
#from paralelMultiHmm import MultipleHMM
from multihmm import MultipleHMM
from sklearn.metrics import accuracy_score
import numpy as np
from time import time
from ParamParser import ParamParser
from fileParser import *

from options.HMMOptions import HMMOptions

def run(folderPath, hmm_params, features_params,filename_pattern):
    # Remove the full path -> get the raw dataset name
    dataset = folderPath.split('-')[0]
    dataset = dataset.split('/')[1]

    # Set labelsPath file to get the real music genre
    labelsPath = folderPath + '/' + dataset + '_labels.txt'

    output = open(folderPath + '/output.txt', 'w')
    output.write('n_components,covariance_type,n_iter,features,classifier,nfeats,n_frames,selector,f1,stdev,time_elapsed\n')

    # Get genre to int dictionary
    genre_dict = generateGenreDict(labelsPath)

    for hmm_ncomponents in hmm_params['n_components']:
        for hmm_convariancetype in hmm_params['covariance_type']:
            for hmm_niter in hmm_params['n_iter']:

                hmm = MultipleHMM(GaussianHMM(hmm_ncomponents, hmm_convariancetype, hmm_niter))

                # Initialize train and tests vectors

                for current_frame in features_params['frames']:
                    # print "Frame: {}".format(current_frame)
                    for current_hs in features_params['hs']:
                        # print "HS: {}".format(current_hs)

                        fold_accuracys = []
                        start_time = time()
                        for test_fold in range(1, features_params['folds_number'] + 1):

                            train_X, train_Y, test_X, test_Y = [], [], [], []
                            train_length = []

                            fileName = filename_pattern.format(
                                folderPath, current_frame, current_hs, test_fold)

                            try:
                                test_X, test_Y, test_train_length = extractFeaturesAndLabelFromFile(
                                    fileName, labelsPath, genre_dict)
                            except Exception as e:
                                writeNotFound(fileName)
                                print(e)
                                raise e

                            for current_fold in range(1, features_params['folds_number'] + 1):
                                if current_fold == test_fold:
                                    continue

                                fileName = filename_pattern.format(
                                    folderPath, current_frame, current_hs, current_fold)

                                try:
                                    train_X_tmp, train_Y_tmp, train_length_tmp = extractFeaturesAndLabelFromFile(
                                        fileName, labelsPath, genre_dict)

                                except:
                                    writeNotFound(fileName)
                                    print('File not found: {}'.format(fileName))

                                train_length = train_length + train_length_tmp

                                # train_X = np.array(train_X)
                                train_X = train_X + train_X_tmp

                                train_Y = train_Y + train_Y_tmp
                                # Concatena vetores normais obtidos do extractFeaturesAndLabelFrinFile
                                # Vetores obtidos tem matrizes numpy de shape (-1,1)
                                # train_X_tmp, train_Y_tmp representam caracteristicas extraídas de um único arquivo
                                # train_X, train_Y representam características estraidas de n folds (total)

                            # Transforma vetores normais em numpy arrays
                            train_length = np.array(train_length)
                            train_X = np.array(train_X)
                            train_Y = np.array(train_Y)

                            hmm.fit(train_X, train_Y, train_length)

                            test_X = np.array(test_X)
                            classified = []

                            for i in test_X:
                                classified.append(hmm.predict(i))
                            fold_accuracys.append(accuracy_score(test_Y, classified))


                        elapsed = time() - start_time
                        average = np.mean(np.array(fold_accuracys))
                        stdev = np.std(np.array(fold_accuracys))

                        feature = folderPath.split('/')[1].split('_')[1]

                        output.write("{},{},{},{},{},{},{},{},{},{}\n".format(hmm_ncomponents, hmm_convariancetype, hmm_niter,feature,'SVM', current_hs, current_frame, average, stdev, elapsed))
                        print("{},{},{},{},{},{},{},{},{},{}\n".format(hmm_ncomponents, hmm_convariancetype, hmm_niter,feature,'SVM', current_hs, current_frame, average, stdev, elapsed))
                    output.write(',,,,,,,,,,,\n')


def writeNotFound(file):
    with open('missingFiles.txt', 'a') as missingFiles:
        missingFiles.write(file + '\n')

def main():
    cli_options = HMMOptions().get_parser().parse_args()
    parser = ParamParser(cli_options.input)

    run(cli_options.input, parser.hmm_params, parser.feature_params, parser.filename_pattern)


if __name__ == '__main__':
    main()
