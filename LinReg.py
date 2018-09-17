
import random
import numpy as np
from numpy import *
import argparse
import sys


parser = argparse.ArgumentParser(description='Predict using Linear Regression')
parser.add_argument('data', metavar='data', type=str, nargs=1, help='data to be evaluated/trained on. Data must be in a csv file format')
parser.add_argument('--mode', metavar='train|test', type=str , nargs=1, choices = ['train', 'test'] , help='select the mode (train/test)', required=True)
parser.add_argument('--model', metavar='model', type=str, nargs=1, help='model to test with (will be ignored if train is selected as the mode)')
parser.add_argument('-f', metavar='folds', type=int, default=10, help='number of folds to use for cross-validation. Default is 10')

params = parser.parse_args()
#get the filename
filename = params.data[0]
#get mode as 'train'|'test'
mode = params.mode[0]


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
	#kFolds = int(input('Input the number of folds: '))
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
#				print(str(j) + ": " + str(len(kFeatures[j])))
#				print(str(j) + ": " + str(len(kLabels[j])))
				pass
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

	indexOfLowestLoss = -1
	lowestLoss = 100

	# Fold i wil be the test fold. The rest are training.
	for i in range(kFolds): # Selects the test fold
		trainingFoldFeatures = []
		trainingFoldLabels = []

		for j in range(kFolds): # Goes through all the training folds (i.e. the rest of the folds)
			if i == j: # Don't train using the test fold.
				continue
			for q in range(len(npkFeatures[j])):
				trainingFoldFeatures.append(npkFeatures[j][q]) # Form the training set from the folds
				trainingFoldLabels.append(npkLabels[j][q])

		npFeatures = np.array(trainingFoldFeatures, dtype=float)
		npFeatures = np.matrix(npFeatures, dtype=float)
		tFeatures = npFeatures.getT()
		try:
			npLabels = np.array(trainingFoldLabels, dtype=int)
		except:
			print(trainingFoldLabels)
			sys.exit(0)
		npLabels = npLabels.reshape((-1, 1))

		# compute the weight vector for linear regression.
		# compute the a matrix as the matrix of the sample set times it's transposition
		A = tFeatures.dot(npFeatures)
		# compute the b vector as the sample matrix times the label set
		b = tFeatures.dot(npLabels)

		#np.seterr(all='ignore')
		np.warnings.filterwarnings('ignore')
		# get the inverse of the matrix
		try:
			Ainv = np.linalg.inv(A)
		except np.linalg.LinAlgError:
			# weight = np.linalg.lstsq(A, b, rcond=None)[0]
			(u, d, v) = np.linalg.svd(A, full_matrices=True, compute_uv=True)
			v = v.getH()
			d = np.diag(d)
		#	print(d)
			d = 1. / d
		#	print(d)
			d = np.diag(np.diag(d))
		#	print(d)
			u = u.getT()

			temp = d.dot(u)
			Ainv = v.dot(temp)

		kWeights[i] = Ainv.dot(b)

		loss = 0
		for j in range(len(npFeatures)):
			loss += 1/len(npFeatures) * (npFeatures[j].dot(kWeights[i]) - npLabels[j])**2 #loss on the training as 1/m sumof (<w_i, x_i> - y_i)^2
		if loss < lowestLoss:
			lowestLoss = loss
			indexOfLowestLoss = i
#		print(loss)
		del trainingFoldFeatures
		del trainingFoldLabels


	with open("linregmodel.csv", 'w') as f:
		for i in range(len(kWeights[indexOfLowestLoss])):
			if i < len(kWeights[indexOfLowestLoss]) - 1:
				f.write('{}, '.format(kWeights[indexOfLowestLoss][i]).replace('[', '').replace(']', ''))
			else:
				f.write('{}\n'.format(kWeights[indexOfLowestLoss][i]).replace('[', '').replace(']', ''))


# Test Empirical Loss here:

def test():
	# Regular python arrays used to hold feature and label values.
	features = []
	labels = []

	# Process file input here
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
	npFeatures = np.matrix(npFeatures, dtype=float)
	npLabels = np.array(labels, dtype = int)

	#print('Testing the loss here.')
	#print('There are ' + str(len(npFeatures)) + ' test points.')

	# This covers square loss. We still want to see the absolute loss in terms of how many points are classified correctly or incorrectly.
	loss = 0
	for i in range(len(npFeatures)):
			loss += 1/len(npFeatures) * (npFeatures[i].dot(weight) - npLabels[i])**2 #loss on the training as 1/m sumof (<w_i, x_i> - y_i)^2

	print('The linear regression squared loss is ' + str(loss * 100.0) + '%.')


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
		

		# value = npFeatures[i].dot(weight)
		# #print(value)
		# if abs(-1 - value) > abs(1 - value):
		# 	value = 1
		# else:
		# 	value = -1

		# if value == npLabels[i]:
		# 	numCorrect += 1
		# else:
		# 	numWrong += 1


	print('You were correct on ' + str(numCorrect) + ' of the ' + str(len(npFeatures)) + ' test points.')
	print('You were wrong on ' + str(numWrong) + ' of the ' + str(len(npFeatures)) + ' test points.')

	# print('\nThis gives you a success rate of ' + str(numCorrect / len(npFeatures) * 100.0) + '%.')
	# print('This gives you a loss rate of ' + str(numWrong / len(npFeatures) * 100.0) + '%.')


	precision = (tp / (tp+fp))
	recall = (tp / (tp+fn))
	f1 = (2*(precision*recall / (precision+recall)))

	print("Your accuracy was " + str(numCorrect / len(npFeatures) * 100.0) + '%.')
	print('Your recall was ' + str(recall * 100.0) + '%.')
	print('Your precision was ' + str(precision * 100.0) + '%.')
	print('Your F1 score was ' + str(f1 * 100) + '%.')


if mode == 'test':
	test()
elif mode == 'train':
	train()