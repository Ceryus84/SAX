"""
Aidan Butcher
File for custom functions to create a pandas Dataframe and to collect n
random samples for the training set.
Todo: None.
Bugs: None.
"""


import os, random
import pandas as pd
import numpy as np
import constants


# constants for preprocessing
FILE_DIR = constants.FILE_DIR
EXTENSION = constants.EXTENSION
TRAIN_FILE = constants.TRAIN_FILE
NUM_FILES = constants.NUM_FILES
LOG = constants.LOG
NUM_FEATS = constants.NUM_FEATS

# checks to make sure NUM_FEATS is valid
if isinstance(NUM_FEATS, int):
    if NUM_FEATS < 0 or NUM_FEATS > 2:
        raise ValueError("NUM_FEATS must be 1 or 2.")
else:
    raise TypeError("NUM_FEATS must be an integer of value 1 or 2.")


def df_parser(dir: str, ext: str) -> pd.DataFrame:
    """
    Creates a pandas Dataframe from directory, sorting into 4 columns.
    dir : the string for the directory name
    ext : the string for the file extension
    """
    # set up pandas Dataframe and columns
    col = ['Filename', 'Sample Name', 'Current Time', 'Total Time']
    f_and_t = pd.DataFrame(columns=col)

    # iterates through dir and store all relevant values in pd Dataframe
    for row, filename in enumerate(os.listdir(dir)):
        # non-specified file extensions are ignored
        if filename.endswith(f'{ext}'):
            # uses split to get all info separated by underscores
            res = filename[:-4].split('_')
            # store filename, sample name, current time and predicted time in pd.Dataframe
            f_and_t.loc[row, col[0]] = filename
            f_and_t.loc[row, col[1]] = res[0]
            f_and_t.loc[row, col[2]] = int(res[1])
            f_and_t.loc[row, col[3]] = int(res[2])
    
    return f_and_t


def file_splicer(df: pd.DataFrame, filename1:
                 str, filename2: str, dir: str,
                 feats: int = 1) -> list[np.ndarray, np.ndarray]:
    """
    Takes 2 files from a directory and splices them to be prepared for
    training/testing sets. Divides by qError for x_train.
    df : pandas DataFrame object holding files
    filename1 : name of the first file that is being spliced
    filename2 : name of the second file that is being spliced
    dir : name of the directory holding the files
    """
    # stores the times of the different files for time factor feature
    f_ct = df.loc[df['Filename'] == filename1]['Current Time'].values[0]
    s_ct = df.loc[df['Filename'] == filename2]['Current Time'].values[0]
    f_n = s_ct / f_ct

    # load and store files into numpy arrays from file
    first = np.loadtxt(os.path.join(dir, filename1), delimiter=',', dtype=float)
    second = np.loadtxt(os.path.join(dir, filename2), delimiter=',', dtype=float)

    # store the quotient array of the intensity column by the qError column
    div_f = np.divide(first[:, 1], first[:, 2])

    # store quotient array with time factor
    if feats == 2:
        new_f = np.concatenate((div_f, first[:, 2], [f_n]))
    else:
        new_f = np.concatenate((div_f, [f_n]))

    # store intensity column of second file
    new_s = second[:, 1]
    return new_f, new_s


def log_splicer(df: pd.DataFrame, filename1:
                 str, filename2: str, dir: str,
                 feats: int = 1) -> list[np.ndarray, np.ndarray]:
    """
    Takes 2 files from a directory and splices them to be prepared for
    training/testing sets. Uses log10 for x_train.
    df : pandas DataFrame object holding files
    filename1 : name of the first file that is being spliced
    filename2 : name of the second file that is being spliced
    dir : name of the directory holding the files
    """
    # stores the times of the different files for time factor feature
    f_ct = df.loc[df['Filename'] == filename1]['Current Time'].values[0]
    s_ct = df.loc[df['Filename'] == filename2]['Current Time'].values[0]
    f_n = s_ct / f_ct

    # load and store files into numpy arrays from file
    first = np.loadtxt(os.path.join(dir, filename1), delimiter=',', dtype=float)
    second = np.loadtxt(os.path.join(dir, filename2), delimiter=',', dtype=float)

    # prepares Intensity column for log10
    first[:, 1][first[:, 1] <= 0] = 10**-6
    second[:, 1][second[:, 1] <= 0] = 10**-6

    # applies log10 to first and second array
    log_f = np.log10(first[:, 1])
    log_s = np.log10(second[:, 1])

    # store log array with time factor
    if feats == 2:
        new_f = np.concatenate((log_f, first[:, 2], [f_n]))
    else:
        new_f = np.concatenate((log_f, [f_n]))

    # store intensity column of second file
    new_s = log_s
    return new_f, new_s


def feature_array(ft_df: pd.DataFrame,
                  rows: int, num_files: int, dir: str,
                  log: bool, feats: int) -> list[np.ndarray, np.ndarray, list]:
    """
    Picks num_files random x, y pairs of files from each sample and creates a
    training set of x_train and y_train, then stores them in a file.
    ft_df : the pandas DataFrame containing everything.
    rows : amount of rows the arrays need to have
    num_files : number of files collected from each sample
    dir : name of the directory holding the files
    """
    # store unique sample names in list
    u_samples = ft_df['Sample Name'].unique()
    num_sam = len(u_samples)

    # create ndarrays for storing the file data in single columns
    # arr1 is x_train and arr2 is y_train
    arr1 = np.ndarray((num_files*num_sam, rows*NUM_FEATS+1))
    arr2 = np.ndarray((num_files*num_sam, rows))

    # create empty list for collecting samples
    sample_list = []

    # loop for collecting random samples based on num_samples and num_files
    count = 0
    for name in u_samples:
        temp_df = ft_df.loc[ft_df['Sample Name'] == name]
        x_set = temp_df['Filename'].sample(n=num_files, replace=False, ignore_index=True)
        x_set.name = f'X{count}'
        y_set = temp_df['Filename'].sample(n=num_files, replace=False, ignore_index=True)
        y_set.name = f'Y{count}'
        sample_list.append(x_set)
        sample_list.append(y_set)
        count += 1
    
    train_df = pd.concat(sample_list, axis=1)
    # print(train_df)
    count2 = 0
    for i in range(count):
        x = train_df[f'X{i}']
        y = train_df[f'Y{i}']
        for i in range(num_files):
            # create index for storing into arr1 and arr2
            index = i + (num_files * count2)

            # modifies files to be stored for the training set
            if log:
                first, second = log_splicer(ft_df, x[i], y[i], dir, feats)
            else:
                first, second = file_splicer(ft_df, x[i], y[i], dir, feats)

            # store single column arrays into arr1 & arr2
            arr1[index] = first
            arr2[index] = second
        count2 += 1
    df1 = pd.DataFrame(arr1)
    df2 = pd.DataFrame(arr2)
    df3 = pd.concat((df1.T, df2.T), axis=1)
    # print(df3.shape)
    df3.to_csv(TRAIN_FILE, sep=',', index=False)


def main():
    # generate random seed
    print("Generating training set.")
    random.seed()
    f_and_t = df_parser(FILE_DIR, EXTENSION)
    rows = np.loadtxt(os.path.join(FILE_DIR, f_and_t['Filename'][0]), delimiter=',', dtype=float).shape[0]
    # print(rows)
    # print(f_and_t.shape)
    feature_array(f_and_t, rows, NUM_FILES, FILE_DIR, LOG, NUM_FEATS)
    print("Training set has been generated.")
    # arr3 = np.loadtxt('Training_Set.csv', dtype=np.float32, delimiter=',')


if __name__ == '__main__':
    """
    Makes sure that nothing in main runs unless this file is run
    specifically.
    """
    main()
