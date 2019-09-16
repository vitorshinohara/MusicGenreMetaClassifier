from params import model_params, layers_params
from fileManager.core import *
from neuralNet import fitAndEvaluate
import sys


def main(datasetPath):
    sequence_length=20
    k = 10
    dataset_params, nameTemplate, datasetName = generateDatasetParams(datasetPath)
    datasetFullName = datasetPath.split('/')[1]
    initializeOutputFile(datasetFullName)

    if datasetName == 'gtzan' or datasetName == 'lmd':
        class_number = 10
    elif datasetName == 'homburg':
        class_number = 9
    elif datasetName == 'exballroom':
        class_number = 13

    print class_number


    labelsPath = dataset_params['labels_path']
    for frame_number in dataset_params['frames_number']:
        for feat_number in dataset_params['n_feats']:
            if feat_number == -1:
                currentNameTemplate = datasetPath + nameTemplate.format(frame_number, '{}')
                feat_number = 26
            else:
                currentNameTemplate = datasetPath + nameTemplate.format(frame_number, feat_number, '{}')

            for rnnLayerEnabled in layers_params['rnnEnabled']:
                for k in layers_params['k']:
                    for k_relu in layers_params['k_relu']:
                        for epoch in model_params['epochs']:
                            for monitor in model_params['monitor']:

                                acc, stdev = fitAndEvaluate(sequence_length, k, k_relu, rnnLayerEnabled, epoch, monitor, class_number, currentNameTemplate, labelsPath,  dataset_params['fold_number'])
                                write(datasetFullName, '{},{},{},{},{},{},{},{},{}\n'.format(frame_number,feat_number,k,k_relu,rnnLayerEnabled,epoch,monitor,acc,stdev))


def generateDatasetParams(datasetPath):

    datasetName = datasetPath.split('-')[0].split('/')[1]
    featureExtractorName = datasetPath.split('/')[1].split('_')[1]

    n_feats, nameTemplate = getNumberOfFeats(featureExtractorName)

    dataset_params = {
        'labels_path' : '{}{}_labels.txt'.format(datasetPath, datasetName),
        'fold_number' : 3 if 'lmd' in datasetName else 10,
        'frames_number' : [5, 20] if 'exballroom' in datasetName else [5, 20, 40],
        'n_feats' : n_feats
    }

    return dataset_params, nameTemplate, datasetName


def getNumberOfFeats(featureExtractorName):
    n_feats = [-1]
    nameTemplate = ''

    if 'ae-feats' in featureExtractorName:
        n_feats = [64, 128, 256]
        nameTemplate = '{}-nf_linspace-fs_{}-hs_{}-f.predict.frames.txt'

    elif 'ae16,32-feats' in featureExtractorName:
        n_feats = [16, 32]
        nameTemplate = '{}-nf_linspace-fs_{}-hs_{}-f.predict.frames.txt'

    elif 'gtzan-feats' in featureExtractorName:
        n_feats = [-1]
        nameTemplate = '{}-nf_linspace-fs_{}-f.predict.frames.txt'

    elif 'mel-feats' in featureExtractorName:
        n_feats = [128]
        nameTemplate = '{}-nf_linspace-fs_{}-nfeats_none-nl_True-noanova_True-delta_{}-f.predict.frames.txt'

    elif 'rp-feats' in featureExtractorName:
        n_feats = [8, 26, 51, 76, 100]
        nameTemplate = '{}-nf_linspace-fs_{}-nfeats_none-nl_True-noanova_True-delta_{}-f.predict.frames.txt'
    return n_feats, nameTemplate

if __name__ == '__main__':
    datasetPath = sys.argv[1]
    main(datasetPath)