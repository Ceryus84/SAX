"""
Aidan Butcher
Second attempt at making a model for the processing of data for the project.
Attempting to implement a Bayesian NN for this program along with a custom
loss function.
Bugs: Can view mean or stdv of model output, but not much else. Model is
      not running properly, only returning nan for loss, mean and stdv of
      output.
Todo: Fix BNN to run properly, custom loss function, get consistent
      set of data. Set prior and posterior mean with gamma distribution, where
      mode is 0 and stdv is 1 and try to get working.
"""


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras as ks
import tf_keras
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from Filename_Parsing import df_parser, file_splicer
import constants

tfd = tfp.distributions

# constants for the file
FILE_DIR = constants.FILE_DIR
TRAIN_FILE = constants.TRAIN_FILE
EXTENSION = constants.EXTENSION
LEARN_RATE = constants.LEARN_RATE
EPOCHS = constants.EPOCHS
BATCH_SIZE = constants.BATCH_SIZE
VAL_SPLIT = constants.VAL_SPLIT
TEST_FILE_X = constants.TEST_FILE_X
TEST_FILE_Y = constants.TEST_FILE_Y

# get number of unique samples
ft_df = df_parser(FILE_DIR, EXTENSION)
u_samples = ft_df['Sample Name'].unique()
num_samples = len(u_samples)

# read in train_df and store num of cols and rows
train_df = pd.read_csv(TRAIN_FILE)
num_cols = train_df.shape[1]
num_rows = train_df.shape[0]

# split train_df into x_train & y_train DataFrames
x_train = train_df.iloc[:, :num_cols//2]
y_train = train_df.iloc[:num_rows-1, num_cols//2:]

# transpose and convert DataFrames to numpy arrays
x_train = x_train.T.to_numpy()
y_train = y_train.T.to_numpy()

# amount_rows_x is input shape & amount_rows_y is output shape
amount_rows_x = x_train.shape[1]
amount_rows_y = y_train.shape[1]

# define negative log-likelood, which is used as a loss function of sorts
# not entirely sure how it works, but has not errored since changes
negloglik = lambda y, p_y: -p_y.log_prob(y)


def posterior(kernel_size, bias_size=0, dtype=None):
    """
    Attempt at creating a posterior mean function using the Gamma distribution.
    Not sure if mode or stddev is right, but currently just trying
    to get something to work.
    """
    n = kernel_size + bias_size
    c = np.log(np.exp(1.))
    return tf_keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Gamma(concentration=1.0, rate=c * t)),
    ])


def prior(kernel_size, bias_size=0, dtype=None):
    """
    Attempt at creating a prior mean function using the Gamma distribution,
    Not sure if mode or stddev is right, but currently just
    trying to get something to work.
    """
    n = kernel_size + bias_size
    return tf_keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Gamma(concentration=1.0, rate=t)),
    ])


# bnn implementation using a tf_keras Sequential model for convenience (WIP)

# MODEL IS RUNNING! WOOHOO!
model = tf_keras.Sequential([
    tf_keras.layers.Input(shape=(amount_rows_x,)),
    tfp.layers.DenseVariational(
        units=494,
        kl_weight=1/num_samples,
        activation='relu',
        make_posterior_fn=posterior,
        make_prior_fn=prior,
    ),
    tfp.layers.DistributionLambda(lambda t: tfd.Gamma(concentration=2.0, rate=t))
])

# compile model, set loss and optimizer
model.compile(
    loss=negloglik,
    optimizer=tf_keras.optimizers.Adam(learning_rate=LEARN_RATE),
    metrics=['accuracy']
)

# trains the model and stores the stats in history for potential observation
model.fit(x=x_train, y=y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
# print(history.history)

# load test file
filename1 = TEST_FILE_X
filename2 = TEST_FILE_Y
file = os.path.join(FILE_DIR, filename1)
file_f = os.path.join(FILE_DIR, filename2)
x_test_twocol = np.loadtxt(file, delimiter=',', dtype=float)
y_test_twocol = np.loadtxt(file_f, delimiter=',', dtype=float)

first_cval = filename1[:-4].split('_')
first_cval = int(first_cval[1])
second_cval = filename2[:-4].split('_')
second_cval = int(second_cval[1])
n_test = second_cval / first_cval

# prepare test files for model prediction
x_test, y_test = file_splicer(ft_df, filename1, filename2, FILE_DIR)
x_test_twocol[:, 1][x_test_twocol[:, 1] <= 0] = 10**-6

"""
Model is currently training and making predictions, however yhat is
not working with plotting. Need to figure out how to modify yhat such that it
can be plotted and verified.
"""
# model predicts on x_test file
yhat = model(np.array([x_test]))
# current output is a tfp.distributions instance, which so far cannot be easily
# plotted, need to continue reading tfp API
print(yhat.mean())

# plot x, y_true and y_pred for evaluation purposes
plt.plot(x_test_twocol[:, 0], np.log10(x_test_twocol[:, 1]), c='g', label='Original')

# yhat[yhat <= 0] = 10**(-6)
plt.plot(x_test_twocol[:, 0], np.log10(yhat), c='r', label='Predicted')
plt.plot(x_test_twocol[:, 0], yhat, c='r', label='Predicted')
plt.plot(y_test_twocol[:, 0], np.log10(y_test_twocol[:, 1]), 'b', label='Actual')
plt.xlabel("q (Å)")
plt.ylabel("Intensity")
plt.title("Intensity vs. q")
plt.legend()
plt.show()
print(yhat)
