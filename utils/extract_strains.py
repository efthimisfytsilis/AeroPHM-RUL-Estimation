import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt

def extract_strains(file, change_points=None):
    '''Loads the strain measurements from given file path, splits the data into QSs, 
    constructs virtual FBGs, namely 15 at each stiffeners foot and calculates the average strain

    Parameters:
    - file (str): path to the .txt file containing the strain measurements 
    - change_points (list): list of indices where the change points of the measuring tape are located, 12 in number are expected

    Returns:
    - strains (pd.DataFrame): average strain for each vFBG over 1/3 of sample length of each QS, shape (n_QS, 15*6 = 90)
    - fig1 (plt.figure): figure showing the QSs separation
    - fig2 (plt.figure): figure showing the change points
    - change_points (list): list of indices where the change points are located
    '''
    # Initial strain measurements loading
    total_columns = pd.read_csv(file, skiprows=5, nrows=0, sep='\t').shape[1]
    data = pd.read_csv(file, skiprows=5, sep='\t', usecols=list(range(1,total_columns)), header=None, dtype='float32')
    
    ## Extract x axis
    increment= pd.read_csv(file, skiprows=4, sep='\t', nrows=1, header=None).iloc[0][1:]
    
    ## Extract time axis
    timestamp = pd.read_csv(file, skiprows=5, usecols=[0], sep='\t', header=None)
    timestamp = pd.to_datetime(timestamp[0])
    timestamp = (timestamp - timestamp[0]).dt.total_seconds()

    # Find where each QS happens
    QS_idx = timestamp.diff()
    QS_idx = QS_idx[QS_idx > 20].index.tolist()

    ### QS separation check
    fig1, ax1 = plt.subplots()
    ax1.vlines(QS_idx, ymin=0, ymax=max(timestamp), color='r')
    ax1.plot(timestamp)
    ###

    ## Splitting the dataframe into QSs based on QS_index
    QSs = []
    if QS_idx[0] < 10:
        QS_idx.pop(0)
    for i in range(len(QS_idx)):
        if i == 0:
            QSs.append(data.iloc[:QS_idx[i], :])
        else:
            QSs.append(data.iloc[QS_idx[i-1]:QS_idx[i], :])
    QSs.append(data.iloc[QS_idx[-1]:, :])
    ## Remove QSs that are too short
    QSs = [QS for QS in QSs if QS.shape[0] > 10]

    # Find where measurements are at the stiffeners foot (change points)
    if change_points is None:
        ## Take the last of 20 random samples from the 2nd to last QS
        sample = QSs[-2].sample(20).sort_index().iloc[-1]
        
        ## Smooth the sample
        sample = sample.rolling(window=50).mean()
        sample.fillna(sample.dropna().iloc[0], inplace=True)
        
        ## Find 12 change points
        algo = rpt.Binseg(model='l2').fit(sample.values.T.reshape(-1,1))
        change_points = algo.predict(n_bkps=12)[:-1]

        ### Change point check
        fig2, ax2 = plt.subplots()
        ax2.vlines(change_points, ymin=min(sample), ymax=max(sample), color='r', linestyle='--')
        ax2.plot(sample)
        ###

        # Virtual FBGs construction
        ## Calculate the measurement leangth in each foot
        measurement_length = [change_points[i + 1] - change_points[i] for i in range(0, len(change_points) - 1, 2)] 
        
        ## Make sure the length is the same for all feet
        difference = [measurement - min(measurement_length) for measurement in measurement_length]
        ### Equally split the resulting difference to the left and right of the change point
        difference = [int(np.floor(i/2)) for i in difference]
        difference = [item for item in difference for _ in range(2)]
        change_points = [x + y if i % 2 == 0 else x - y for i, (x, y) in enumerate(zip(change_points, difference))]
    
    # --------------------------------------------------------------------------------------------
    # ## Measurement every inc [mm]
    # inc = increment.iloc[1]-increment.iloc[0]

    # ## Convert start, step from mm to index
    # start = int(40/inc)
    # step = int(50/inc)
    
    # ## Create a list of slices based on the column indices
    # slices = [slice(change_points[i] + start, change_points[i+1]+1, step) for i in range(0, len(change_points)-1, 2)]
    
    # # Strains for 15 vFBGs at each foot
    # strain = []
    # for qs in QSs:
    #     df_slices = [qs.iloc[:, s] for s in slices]
    #     for i, df_slice in enumerate(df_slices):
    #         if len(df_slice.columns) > 15:
    #             df_slices[i] = df_slice.iloc[:, :15]
    #     strain_df = pd.concat(df_slices, axis=1)
    #     strain.append(strain_df)
    # --------------------------------------------------------------------------------------------
    # or create a list of vFBGs for each foot
    # --------------------------------------------------------------------------------------------
    vfbg_list = [np.linspace(change_points[i], change_points[i+1], num=15) for i in range(0, len(change_points) - 1, 2)]
    vfbg_flat = [int(vfbg) for foot in vfbg_list for vfbg in foot]
    strain = []
    for qs in QSs:
        df_qs = qs.loc[:, vfbg_flat]
        strain.append(df_qs)
    # --------------------------------------------------------------------------------------------
        
    # Calculate the average strain for each vFBG over 1/3 of sample length of each QS
    fraction = 1/3
    strains = pd.DataFrame()
    for qs in strain:
        num_samples = round(fraction*len(qs))
        choice = np.random.choice(qs.index, size=num_samples, replace=False)
        qs_mean = pd.Series(np.nanmean(qs.loc[choice], axis=0))
        strains = pd.concat([strains, qs_mean], axis=1, ignore_index=True)
    
    ## or over the whole QS
    # strain = [s.mean(axis=0) for s in strain]
    # strain = pd.concat(strain, axis=1)
    strains = strains.T
    
    return strains, change_points