"""
Aidan Butcher
First attempt at making a model for the processing of data for the project.
Will likely not be the last attempt, but depends on results.
Todo: 1. Re-normalize the data for the plotting of the non-LOG version
      2. Additionally attempt to run the non-LOG data on the LOG model.
Bugs: A potential division by zero with the non-LOG version of the model
      that might be creating issues.
"""


import os, random
import numpy as np
import pandas as pd
import keras as ks
import matplotlib.pyplot as plt
from Filename_Parsing import df_parser, file_splicer, log_splicer, main
import constants


# constants for directory and number of files
FILE_DIR = constants.FILE_DIR
EXTENSION = constants.EXTENSION
TRAIN_FILE = constants.TRAIN_FILE

# constants for model training
EPOCHS = constants.EPOCHS
BATCH_SIZE = constants.BATCH_SIZE
LEARN_RATE = constants.LEARN_RATE
VAL_SPLIT = constants.VAL_SPLIT
NUM_FILES = constants.NUM_FILES
EXTRA_LAYER = constants.EXTRA_LAYER
LOG = constants.LOG
NUM_FEATS = constants.NUM_FEATS

# constants for model testing
TEST_FILE_X = constants.TEST_FILE_X
TEST_FILE_Y = constants.TEST_FILE_Y

# checks to make sure that NUM_FEATS is valid
if isinstance(NUM_FEATS, int):
    if NUM_FEATS < 0 or NUM_FEATS > 2:
        raise ValueError("NUM_FEATS must be 1 or 2.")
else:
    raise TypeError("NUM_FEATS must be the ints 1 or 2.")

# will create a training set if one does not already exist
try:
    train_df = pd.read_csv(TRAIN_FILE)
    print("Training file found, proceeding to model compilation and training.")
except FileNotFoundError as error:
    print("Training file not found, creating file with current parameters.")
    main()
    print("Moving to model compilation and training.")
    train_df = pd.read_csv(TRAIN_FILE)

num_cols = train_df.shape[1]
num_rows = train_df.shape[0]
# print(f'Rows: {num_rows}, Columns: {num_cols}')

# split train_df into x_train & y_train DataFrames
x_train = train_df.iloc[:, :num_cols//2]
if NUM_FEATS == 2:
    y_train = train_df.iloc[:(num_rows-1)//2, num_cols//2:]
else:
    y_train = train_df.iloc[:num_rows-1, num_cols//2:]

# transpose and convert DataFrames to numpy arrays
x_train = x_train.T.to_numpy()
y_train = y_train.T.to_numpy()

# amount_rows_x is input shape & amount_rows_y is output shape
amount_rows_x = x_train.shape[1]
amount_rows_y = y_train.shape[1]
print(x_train.shape, y_train.shape)

# mlp functional implementation
inputs = ks.Input(shape=(amount_rows_x,))
dense = ks.layers.Dense(units=100, activation='relu')
x = dense(inputs)
if EXTRA_LAYER:
    # if true adds an additional layer to the model architecture
    x = ks.layers.Dense(units=100, activation='relu')(x)
outputs = ks.layers.Dense(units=amount_rows_y)(x)
model = ks.Model(inputs=inputs, outputs=outputs)

# compile model, set loss, optimizer and metrics
model.compile(
    loss=ks.losses.MeanSquaredError(),
    optimizer=ks.optimizers.Adam(learning_rate=LEARN_RATE),
    metrics=[ks.metrics.MeanSquaredError()]
)

# trains the model and stores the stats in history for potential observation
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VAL_SPLIT)
# print(history.history)

# load test file
filename1 = TEST_FILE_X
filename2 = TEST_FILE_Y
file = os.path.join(FILE_DIR, filename1)
file_f = os.path.join(FILE_DIR, filename2)
x_test_twocol = np.loadtxt(file, delimiter=',', dtype=float)
y_test_twocol = np.loadtxt(file_f, delimiter=',', dtype=float)

# prepare test file
first_cval = filename1[:-4].split('_')
first_cval = int(first_cval[1])
second_cval = filename2[:-4].split('_')
second_cval = int(second_cval[1])
n_test = second_cval / first_cval

# prepare test file for model testing
x_test_twocol[:, 1][x_test_twocol[:, 1] <= 0] = 10**-6
ft_df = df_parser(FILE_DIR, EXTENSION)
if LOG:
    x_test, y_test = log_splicer(ft_df, filename1, filename2, FILE_DIR, NUM_FEATS)
else:
    x_test, y_test = file_splicer(ft_df, filename1, filename2, FILE_DIR, NUM_FEATS)

# store prediction and prepare it for plotting
yhat = model.predict(np.array([x_test]))
yhat = yhat.T
# print(yhat)

# plot x_test, y_test and y_pred
plt.plot(x_test_twocol[:, 0], np.log10(x_test_twocol[:, 1]), c='g', label='5 seconds')
if LOG:
    plt.plot(x_test_twocol[:, 0], yhat, c='r', label='Predicted')
else:
    yhat[yhat <= 0] = 10**(-6)
    plt.plot(x_test_twocol[:, 0], np.log10(yhat), c='r', label='Predicted')
plt.plot(y_test_twocol[:, 0], np.log10(y_test_twocol[:, 1]), 'b', label='50 seconds')
plt.xlabel("q (Å)")
plt.ylabel("Intensity")
plt.title("Intensity vs. q")
plt.legend()
plt.show()
# print(yhat, y_test)
