import numpy as np
import pandas as pd

def load_data(filename, smooth=True, threshold=None):
    '''Loads the data from the given file path for both L1 and L2 specimens, adds the RUL column and splits
    the data into training and testing sets
      
    Parameters:
    - filename (str): path to the .xlsx file containing the data, where each sheet is a specimen
    - smooth (bool): whether to smooth the hi_ga column or not
    - threshold (float): the threshold value to crop the hi_ga series at
      
    Returns:
    - Xtrain (dict): dictionary where the keys are the specimen names and the values are the corresponding hi_ga sequences
    - Ytrain (dict): dictionary where the keys are the specimen names and the values are the corresponding RULs
    - Xtest (dict): smoothed hi_ga sequences for the test specimens
    - Ytest (dict): 
    - [train_specimens, test_specimens] (list): list containing the names of the specimens used for training and testing'''
    # Load the data
    all_sheets = pd.read_excel(filename, sheet_name=None, header=None, names=['cycles', 'hi_ga'])
    
    # Specify which specimens to use for training and testing
    train_specimens = [item for item in all_sheets.keys() if item.startswith(('ca','va','sp'))]
    test_specimens = [item for item in all_sheets.keys() if item not in train_specimens]

    # Crop training series when the HI first reaches the 0.845 threshold and smooth the hi_ga column
    for specimen in test_specimens:
        if smooth:
            all_sheets[specimen].hi_ga = all_sheets[specimen].hi_ga.rolling(all_sheets[specimen].hi_ga.shape[0]//10, min_periods=1).mean()
        if threshold:
            all_sheets[specimen] = all_sheets[specimen].loc[all_sheets[specimen].hi_ga <= threshold]
    
    # Add RUL column to each dataframe
    for specimen in all_sheets.keys():
        all_sheets[specimen]['rul'] = all_sheets[specimen].cycles.max() - all_sheets[specimen].cycles
        
    # Split the data into training and testing sets
    Xtrain = {specimen: all_sheets[specimen].hi_ga for specimen in train_specimens}
    Ytrain = {specimen: all_sheets[specimen].rul for specimen in train_specimens}
    Xtest = {specimen: all_sheets[specimen].hi_ga for specimen in test_specimens}
    Ytest = {specimen: all_sheets[specimen].rul for specimen in test_specimens}

    return Xtrain, Ytrain, Xtest, Ytest, [train_specimens, test_specimens]