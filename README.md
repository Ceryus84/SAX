# Description
All of the data is stored in the Data folder, so when asked for a directory in any of the files, you should make sure it says Data.
To ensure that everything works properly, keep the python files in the same directory as the Data directory.

# Workflow
Before running First_Model_Attempt.py, make sure that you run Filename_Parsing.py at least once to generate the Training_Set.csv file.
Without this file, the model will error since it does not have the preprocessed data to work with.

# Modifying constants.py
constants.py is the file that contains all constants shared between multiple files. Currently only stores ```LOG``` and ```NUM_FEATS``` constants, which are shared between First_Model_Attempt.py and Filename_Parsing.py. If ```LOG``` = True, then the preprocessing will use log10 instead of dividing by qError. ```NUM_FEATS``` currently only accepts integer values of 1 or 2, if 2 is chosen than preprocessing will include the qError as a feature for the training set.

# Modifying Preprocessing
Filename_Parsing.py is the file for all things preprocessing related. This includes things like randomly collecting n number of samples.
To adjust any of the preprocessing, interact with the constants at the top of Filename_Parsing.py. To change the number of random files collected, modify ```NUM_FILES``` with a positive integer. ```NUM_FILES``` is the amount of files the program will collect from each sample, so if ```NUM_FILES = 100``` and 5 samples, then the result will be 500 files. Samples are automatically detected by the program from the Data directory, so only the amount of files frome each sample is necessary.

# Necessary Packages:
- Pandas-version: pandas-2.2.2
- Numpy-version: numpy-1.26.4
- Keras-version: keras-3.4.1
- Tensorflow-version: 2.17.0
- Tensorflow_Probability-version: 0.24.0
- Scipy-version: 1.14.0
- Matplotlib-version: 3.9.0
- Along with all the dependencies for the above packages.
