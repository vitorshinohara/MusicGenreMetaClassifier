from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, LSTM
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from fileManager.core import *

from preProcessing.core import extractFeaturesAndLabelFromFile
import numpy as np

from sklearn.preprocessing import StandardScaler

def generateModel(sequence_length, k, k_relu, class_number, SimpleRNNEnabled):
	model = Sequential()

	model.add( Dense(1) )

	if SimpleRNNEnabled:
		model.add( SimpleRNN(k, unroll=True))
	else:
		model.add(LSTM(k, unroll=True))

	model.add( Dense(k_relu, activation='relu', kernel_initializer='glorot_uniform') )
	model.add( Dense(class_number, activation='softmax') )

	return model

def optmizeAndCompileModel(model):
	optimizer = Adam(lr=0.001)
	# decay=1e-2, momentum=0.3, nesterov=True
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	return model

def fitAndEvaluate(sequence_length, k, k_relu, rnnLayerEnabled, epoch_size, monitor, class_number, nameTemplate, labels, foldNumber):
	scores = []
	model = generateModel(sequence_length, k, k_relu, class_number, rnnLayerEnabled)
	model = optmizeAndCompileModel(model)

	for i in range(1,foldNumber + 1):
		X_train = []
		Y_train = []
		X_test = []
		Y_test = []
		for j in range(1,foldNumber + 1):
			if i == j:
				continue
			try:
				X, Y = extractFeaturesAndLabelFromFile(nameTemplate.format(j), labels)
				for ele in X:
					X_train.append(ele)
				Y_train = Y_train + Y

			except:
				errorWrite(nameTemplate.format(j))
				continue

		X_train = np.array(X_train)

		try:
			X, Y = extractFeaturesAndLabelFromFile(nameTemplate.format(i), labels)
			for ele in X:
				X_test.append(ele)
			Y_test = Y_test + Y
		except:
			errorWrite(nameTemplate.format(i))
			continue

		model.fit(np.array(X_train), np.array(Y_train), batch_size=200, epochs=epoch_size, verbose=0, validation_split=0.2, callbacks=[
			ModelCheckpoint(filepath='pesos.txt', monitor=monitor, verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
		])
		try:
			model.load_weights('pesos.txt')
		except:
			initializeOutputFile('Cannot load weights. Filename: {}'.format(nameTemplate.format(i)))
			pass

		score = model.evaluate(np.array(X_test), np.array(Y_test), batch_size=20, verbose=0)
		scores.append(score[1])

	return np.mean(scores), np.std(scores)
