
import random
import numpy as np
from numpy import *
import argparse
import sys

parser = argparse.ArgumentParser(description='Predict using a Perceptron algorithm')
parser.add_argument('data', metavar='data', type=str, nargs=1, help='data to be evaluated/trained on. Data must be in a csv file format')
parser.add_argument('--mode', metavar='train|test', type=str , nargs=1, choices = ['train', 'test'] , help='select the mode (train/test)', required=True)
parser.add_argument('--model', metavar='model', type=str, nargs=1, help='model to test with (will be ignored if train is selected as the mode)')
parser.add_argument('-i', metavar='iterations', type=int, default = 10, help='number of iterations to run. Default is 10')
parser.add_argument('-f', metavar='folds', type=int, default=10, help='number of folds to use for cross-validation. Default is 10')

params = parser.parse_args()
#get the filename
filename = params.data[0]
#get mode as train|test
mode = params.mode[0]
iterations = params.i


#if the mode is test but there was no model, exit and explain problem
if mode == 'test' and params.model == None:
	print("--mode test requires a model to be input using --model")
	sys.exit(0)
#if test, get the weight vector
if mode == 'test':
	for line in open(params.model[0]):
	
		line.rstrip('\n')
		line.rstrip('\r')
		x = line.split(',')

		weight = np.array(x, dtype=float)
		#print(weight)


def train():
	file = open(filename)
	# k-fold cross validation
	kFolds = params.f
	kFeatures = []
	kLabels = []
	kWeights = []
	indexOfBestWeight = -1
	accuracyOfBestWeight = 0

	for i in range(kFolds):
		kWeights.append(np.array([0] * 117, dtype = float))

	for i in range(kFolds):
		kFeatures.append([])
		kLabels.append([])

	while True:
		vectors = []
		for i in range(kFolds):
			vectors.append(file.readline())
		random.shuffle(vectors)

		vectors = list(filter(None, vectors))
		tempKFolds = len(vectors)

		if len(vectors) == 0:
			for j in range(kFolds):
				pass
#				print(str(j) + ": " + str(len(kFeatures[j])))
#				print(str(j) + ": " + str(len(kLabels[j])))
#			print('Done getting folds')
			break

		for i in range(tempKFolds):
			vectors[i].rstrip('\n')
			vectors[i].rstrip('\r')
			x = vectors[i].split(',')
			x.append('1.0')
			kFeatures[i].append(x[1:])
			kLabels[i].append(x[0])

	# These are normal lists that contain numpy arrays.
	npkFeatures = []
	npkLabels = []

	for i in range(kFolds):
		npkFeatures.append(np.array(kFeatures[i], dtype=float))
		npkLabels.append(np.array(kLabels[i], dtype = int))

	hasErrors = True
	numIterations = 0

	if kFolds == 1:
		for i in range(kFolds): # Selects the test fold
			trainingFoldFeatures = []
			trainingFoldLabels = []

			for j in range(kFolds): # Goes through all the training folds (i.e. the rest of the folds)
				trainingFoldFeatures.append(npkFeatures[j]) # Form the training set from the folds
				trainingFoldLabels.append(npkLabels[j])

			while hasErrors and numIterations < iterations:
#				print("Iteration: " + str(numIterations) + ", Fold: " + str(i), end='\r')
				numIterations += 1
				hasErrors = False
				for k in range(len(trainingFoldFeatures)): # Go through each fold in the list of training folds
					for p in range(len(trainingFoldFeatures[k])): # Go through elements in each of the individudal training folds
						sign = trainingFoldFeatures[k][p].dot(kWeights[i])

						if (sign >= 0):
							sign = 1
						else:
							sign = -1

						# Update as follows: w(n+1) = w(n) + eta * (d(n) - y(n)) * x(n)
						# d(n) = right answer
						# y(n) = actual response
						temp = (trainingFoldLabels[k][p] - sign)
						kWeights[i] = kWeights[i] + (temp * trainingFoldFeatures[k][p])

						if (trainingFoldLabels[k][p] * sign < 0):
							hasErrors = True
	else:
		# Fold i wil be the test fold. The rest are training.
		for i in range(kFolds): # Selects the test fold
			trainingFoldFeatures = []
			trainingFoldLabels = []

			for j in range(kFolds): # Goes through all the training folds (i.e. the rest of the folds)
				if i == j: # Don't train using the test fold.
					continue
				trainingFoldFeatures.append(npkFeatures[j]) # Form the training set from the folds
				trainingFoldLabels.append(npkLabels[j])

			while hasErrors and numIterations < iterations:
#				print("Iteration: " + str(numIterations) + ", Fold: " + str(i), end='\r')
				numIterations += 1
				hasErrors = False
				for k in range(len(trainingFoldFeatures)): # Go through each fold in the list of training folds
					for p in range(len(trainingFoldFeatures[k])): # Go through elements in each of the individudal training folds
						sign = trainingFoldFeatures[k][p].dot(kWeights[i])

						if (sign >= 0):
							sign = 1
						else:
							sign = -1

						# Update as follows: w(n+1) = w(n) + eta * (d(n) - y(n)) * x(n)
						# d(n) = right answer
						# y(n) = actual response
						temp = (trainingFoldLabels[k][p] - sign)
						kWeights[i] = kWeights[i] + (temp * trainingFoldFeatures[k][p])

						if (trainingFoldLabels[k][p] * sign < 0):
							hasErrors = True
			numIterations = 0
			del trainingFoldFeatures
			del trainingFoldLabels


#	print("Finished training. Now finding best weight vector.")

	for i in range(kFolds):
#		print('There are ' + str(len(npkFeatures[i])) + ' test points.') # Hypothesis i is tested on Fold i and trained on all other Folds.

		numCorrect = 0
		numWrong = 0
		for j in range(len(npkFeatures[i])): # Goes through each element in the test fold, which is Fold i.
			sign = npkFeatures[i][j].dot(kWeights[i])
			if (sign >= 0):
				sign = 1;
			else:
				sign = -1;

			if (sign * npkLabels[i][j] >= 0):
				numCorrect += 1
			else:
				numWrong += 1

		if accuracyOfBestWeight < numCorrect / len(npkFeatures[i]):
			accuracyOfBestWeight = numCorrect / len(npkFeatures[i])
			indexOfBestWeight = i

	# 	print('Testing on Fold ' + str(i + 1) + ' out of ' + str(len(kWeights)) + '.')
	# 	print('You were correct on ' + str(numCorrect) + ' of the ' + str(len(npkFeatures[i])) + ' test points.')
	# 	print('You were wrong on ' + str(numWrong) + ' of the ' + str(len(npkFeatures[i])) + ' test points.')

	# 	print('\nThis gives you a success rate of ' + str(numCorrect / len(npkFeatures[i]) * 100.0) + '%.')
	# 	print('This gives you a loss rate of ' + str(numWrong / len(npkFeatures[i]) * 100.0) + '%.')

	# 	print('\nThe algorithm ran ' + str(numIterations) + ' iterations.\n\n')

	# print("Best Accuracy: " + str(accuracyOfBestWeight))
	# print("Index: " + str(indexOfBestWeight))

	with open("PerceptronModel.csv", 'w') as f:
		for i in range(len(kWeights[indexOfBestWeight])):
			if i < len(kWeights[indexOfBestWeight]) - 1:
				f.write('{}, '.format(kWeights[indexOfBestWeight][i]).replace('[', '').replace(']', ''))
			else:
				f.write('{}\n'.format(kWeights[indexOfBestWeight][i]).replace('[', '').replace(']', ''))



#### END OF CROSS VALIDATION CODE

# Test Empirical Loss here:
def test():
	global weight
	# Regular python arrays used to hold feature and label values.
	features = []
	labels = []

	# Process file input here and 
	for line in open(filename):
		
		# Use rstrip() to get rid of the \n at the end.
		# Need to check to make sure this is the best way of doing it and won't cause errors.
		line.rstrip('\n')
		line.rstrip('\r')
		x = line.split(',')

		# This is meant for adding the bias.
		x.append('1.0')

		features.append(x[1:])
		labels.append(x[0])

	# These are arrays using the Numpy array type. We will pass the regular python features and labels arrays into these.
	npFeatures = np.array(features, dtype=float)
	npLabels = np.array(labels, dtype = int)

	numCorrect = 0
	numWrong = 0
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for i in range(len(npFeatures)):
		sign = npFeatures[i].dot(weight)
		if (sign >= 0):
			sign = 1
		else:
			sign = -1

		if (sign * npLabels[i] >= 0):
			numCorrect += 1
			if(sign > 0):
				tp += 1
			else:
				tn += 1
		else:
			numWrong += 1
			if(npLabels[i] >= 0):
				fn += 1
			else:
				fp += 1

	print('You were correct on ' + str(numCorrect) + ' of the ' + str(len(npFeatures)) + ' test points.')
	print('You were wrong on ' + str(numWrong) + ' of the ' + str(len(npFeatures)) + ' test points.')

	# print('\nThis gives you a success rate of ' + str(numCorrect / len(npFeatures) * 100.0) + '%.')
	# print('This gives you a loss rate of ' + str(numWrong / len(npFeatures) * 100.0) + '%.')

	precision = (tp / (tp+fp))
	recall = (tp / (tp+fn))
	f1 = (2*(precision*recall / (precision+recall)))

	print("{Your accuracy was " + str(numCorrect / len(npFeatures) * 100.0) + '%.')
	print('Your recall was ' + str(recall * 100.0) + '%.')
	print('Your precision was ' + str(precision * 100.0) + '%.')
	print('Your F1 score was ' + str(f1 * 100) + '%.')

if mode == 'train':
	train()
if mode == 'test':
	test()
