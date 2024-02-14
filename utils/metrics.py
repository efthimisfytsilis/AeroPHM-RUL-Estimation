import numpy as np

def monotonicity(hi):
    '''The monotonicity metric of the Health Indicator of a single specimen
    
    Parameters:
    - hi (pd.Series): of shape (n,), where n is the number of time steps
    
    Returns:
    - mon (float): the monotonicity metric
    '''
    return np.abs(np.sign(hi.diff(periods=1)).dropna().sum(axis=0)/hi.shape[0])

def prognosability(hi_start, hi_fail):
    '''The prognosability metric of the Health Indicator based on a population of specimens (test or train)
    
    Parameters:
    - hi_start (np.ndarray): of shape (m,), where m is the number of specimens
    - hi_fail (np.ndarray): of shape (m,), where m is the number of specimens
    
    Returns:
    - prog (float): the prognosability metric
    '''
    return np.exp(-np.std(hi_fail) / np.mean(np.abs(hi_fail-hi_start)))

def cumulative_relative_accuracy(y_true, y_pred):
    '''The Cumulative Relative Accuracy (CRA) metric, we omit the last time step since it is zero
    
    Parameters:
    - y_true (np.ndarray): of shape (n,), where n is the number of time steps
    - y_pred (np.ndarray): of shape (n,), where n is the number of time steps
    
    Returns:
    - cra (float): the CRA metric
    '''
    y_true = y_true[:-1]
    y_pred = y_pred[:-1]
    return np.mean(1-np.abs( (y_true-y_pred) / y_true))

def confidence_interval_distance_convergence(lower, upper):
    '''The Confidence Interval Distance Convergence (CIDC) metric, showcasing whether the confidence interval is converging
    as the amount of data increases during fatigue life
    
    Parameters:
    - lower (np.ndarray): of shape (n,), where n is the number of time steps
    - upper (np.ndarray): of shape (n,), where n is the number of time steps
    
    Returns:
    - cidc (float): the CIDC metric
    '''
    t = np.arange(0, 1000*len(lower), 1000)
    A = 2 * np.sum(np.diff(t)*(upper[:-1] - lower[:-1]))
    Xc = np.sum((t[1:]**2 - t[:-1]**2) * (upper[:-1] - lower[:-1])) / A
    Yc = np.sum(np.diff(t) * (upper[:-1]**2 - lower[:-1]**2)) / A
    return np.sqrt((Xc-t[0])**2 +Yc**2)

def percent_in_bounds(y_true, lower, upper):
    '''The percentage of true values within the confidence interval
    
    Parameters:
    - y_true (np.ndarray): of shape (n,), where n is the number of time steps
    - lower (np.ndarray): of shape (n,)
    - upper (np.ndarray): of shape (n,)
    
    Returns:
    - pib (float): the actual overlap between the true values and the associated bounds'''
    return np.mean((y_true > lower) & (y_true < upper))