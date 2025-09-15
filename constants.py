"""
Aidan Butcher
File for storing constants that are used in multiple files.
Currently stores constants for if log10 should be used instead of quotient
for preprocessing and defines the number of features.
"""

FILE_DIR = 'Data'  # directory storing all of the data used for model train/test
EXTENSION = '.txt'  # file extension of the data being used in training
TRAIN_FILE = 'Training_Set.csv'  # training set file name
EPOCHS = 60  # how many epochs the model will use during training
BATCH_SIZE = 16  # how large of a batch size the model will use for training
LEARN_RATE = 0.001  # determines how high of a learning rate for Adam optimizer
VAL_SPLIT = 0.1  # determines how much of the training set is used for validation
EXTRA_LAYER = False  # determines if the model has 1 or 2 dense layers
LOG = False  # enables/disables using log10 for preprocessing
NUM_FEATS = 1  # only integer values of 1 or 2 are accepted currently
NUM_FILES = 100  # number of files to take from each sample
TEST_FILE_X = 'AgBe_5_1200.txt'  # determines the input for model testing
TEST_FILE_Y = 'AgBe_50_1200.txt'  # determines the expected output for model testing