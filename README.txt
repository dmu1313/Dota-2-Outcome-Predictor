# Dota-2-Outcome-Predictor
ML application developed for CSE 353 at Stony Brook University to predict the winning team of a Dota 2 game based on certain features.
Developed by Daniel Mu along with Jeremy Chu

/////////////////////////////////////////////////////////////////

NAME:
	perceptron.py - Predict using a Perceptron algorithm
	
SYNOPSIS:
	perceptron.py [-h] --mode train|test [--model model] [-i iterations] [-f folds] data

DESCRIPTION:
	Predict using a Perceptron algorithm
	
	Can be used either in training mode or test mode. Training mode will train a model
	and output it to perceptronmodel.csv in the directory where perceptron.py is. Testing
	mode will test the input model on the given data.

	positional arguments:
  	data               data to be evaluated/trained on. Data must be in a csv
                     file format

	optional arguments:
	  -h, --help         show this help message and exit
	  --mode train|test  select the mode (train/test)
	  --model model      model to test with (will be ignored if train is selected
						 as the mode)
	  -i iterations      number of iterations to run. Default value is 10
	  -f folds           number of folds to use for cross-validation. Default value is 10
	  
EXAMPLES:
	python perceptron.py --mode train Train.csv
		will train on 10 folds for 10 iterations each using data from Train.csv.
		
	python perceptron.py --mode train -i 12 -f 8 Train.csv
		will train on 8 folds for 12 iterations each using data from Train.csv.
		
	python perceptron.py --mode test --model perceptronmodel.csv Test.csv
		will evaluate the model on the given test data set.

NAME:
	linreg.py - Predict using Linear Regression
	
SYNOPSIS:
	linreg.py [-h] --mode train|test [--model model] [-f folds] data

DESCRIPTION:
	Predict using Linear Regression
	
	Can be used either in training mode or test mode. Training mode will train a model
	and output it to linregmodel.csv in the directory where linreg.py is. Testing
	mode will test the input model on the given data.
	
	positional arguments:
	  data               data to be evaluated/trained on. Data must be in a csv
						 file format

	optional arguments:
	  -h, --help         show this help message and exit
	  --mode train|test  select the mode (train/test)
	  --model model      model to test with (will be ignored if train is selected
						 as the mode)
	  -f folds           number of folds to use for cross-validation. Default value is 10
	  
EXAMPLES:
	python linreg.py --mode train Train.csv
		will train on 10 folds using data from Train.csv.
		
	python linreg.py --mode train -f 8 Train.csv
		will train on 8 folds using data from Train.csv.
		
	python linreg.py --mode test --model linregmodel.csv Test.csv
		will evaluate the model on the given test data set.