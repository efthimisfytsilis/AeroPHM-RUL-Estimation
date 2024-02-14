import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def hi1(strain):
    '''Evaluates the strain change at current time t relative to
    the reference stage, the early SHM measurements in this case.

    Parameters:
    - strain (pd.DataFrame): strain measurements of shape (n, 90), where n is the number of samples 
    and 90 the number of vFBGs

    Returns:
    - hi (pd.DataFrame): of shape (n, 90)
    '''
    e_ref = strain.iloc[0]
    return np.abs(strain - e_ref) / np.abs(e_ref)

def hi2(strain, n_fbg=15):
    '''Indicates the proportion each vFBG sensor contributes to the cumulative strain among the 15
    vFBG sensors of the same foot
    
    Parameters:
    - strain (pd.DataFrame): strain measurements of shape (n, 90)
    - n_fbg (int): number of vFBG sensors per foot
    
    Returns:
    - hi (pd.DataFrame): of shape (n, 90)
    '''
    # feet = strain.values.reshape(strain.shape[0], n, -1).shape[2]
    columns_multiindex = pd.MultiIndex.from_product([['Foot_'+str(i) for i in range(1, 7)], ['Sensor_'+str(j) for j in range(1, n_fbg+1)]])
    strain = pd.DataFrame(strain.values, columns=columns_multiindex)
    hi = pd.DataFrame()
    for foot in strain.columns.levels[0]:
        hi_foot = strain[foot].div(strain[foot].mean(axis=1), axis=0)
        hi = pd.concat([hi, hi_foot - hi_foot.iloc[0]], axis=1)
    return hi

def hi_fused(hi):
    '''A fusion of HI1 and HI2 for all 90 vFBG sensors, respectively, to obtain a single monotonic 
    HI, employing a weighted summing of the HI values of each vFBG.
    
    Parameters:
    - hi (pd.DataFrame): of shape (n, 90)
    
    Returns:
    - hi_fused (pd.Series): of shape (n,)
    '''
    diff = hi.diff(periods=1).dropna()
    monotonicity = np.abs(((diff>0).sum(axis=0) - (diff<0).sum(axis=0)) / (len(diff) - 1))
    return np.sqrt(np.sum((monotonicity*hi)**2, axis=1))
    
def vhi1(strain, norm_params=None, k=2):
    '''Using PCA, the dimensionality of the available sensor data is decreased
    from 90 to 2. The Euclidian distance is calculated as dL(t) = ‖Z(t) - Z0‖, where Z0=Z(t=0). To 
    normalize the final HI, a radial basis function is used to make sure that it
    begins at 1 and stops working at vHI1f ∈ [ε, ε + δ] with ε = δ = 0.01. The normalizing values in 
    this study are drawn from the historic record generated during the last campaign on single stringered panels
    
    Parameters:
    - strain (pd.DataFrame): strain measurements of shape (n, 90)
    - norm_params (dict): dictionary containing the normalization parameters (z0, dL_min, sl)
    - k (int): number of most similar specimens to the current one
    
    Returns:
    - vhi1 (pd.Series): of shape (n,)
    '''
    pca = PCA(n_components=2)
    # Vector of principal components
    z = pca.fit_transform(strain)
    # Euclidean distance
    dL = np.linalg.norm(z - z[0], axis=1)

    if norm_params is None:
        dL_max = np.max(dL)
        dL_min = np.min(dL[dL>0])
        delta = 0.01
        epsilon = 0.01
        sl = - 0.5 * (dL_max - dL_min)**2 * (1/np.log(delta+epsilon) + 1/np.log(epsilon))
        return np.exp(-(dL-dL_min)**2/sl)

    # Find k most similar specimens to the current one
    z0_dist = np.linalg.norm(norm_params['z0'] - z[0], axis=1)
    specimens = np.argsort(z0_dist)[:k]
    vhi1 = pd.DataFrame()
    for specimen in specimens:
        vhi1 = pd.concat([vhi1, pd.Series(np.exp(-(dL-norm_params['dL_min'][specimen])**2/norm_params['sl'][specimen]))], axis=1)
    return vhi1.mean(axis=1)

def vhi2(strain, strain_healthy=None):
    '''The statistical quantity Q of PCA, also known as the squared sum of residual 
    reconstructed error.
    
    Parameters:
    - strain (pd.DataFrame): strain measurements of shape (n, 90)
    - strain_healthy (pd.DataFrame): strain measurements of shape (n, 90) of the specimen before impact
    if available
    
    Returns:
    - vhi2 (pd.Series): of shape (n,)
    '''
    if strain_healthy is None:
        strain_healthy = strain.iloc[:50]
    scaler = StandardScaler()
    X_healthy = scaler.fit_transform(strain_healthy)
    X = scaler.transform(strain)

    pca = PCA(n_components=2)
    z = pca.fit(X_healthy)
    T = X @ z.components_.T
    X_reconstructed = T @ z.components_
    return pd.Series(np.sum((X - X_reconstructed)**2, axis=1))
    
def higa(hi_3, hi_4, vhi_1, vhi_2):
    '''Combines the above HIs and creates an enhanced one for the SSP histories, with 
    the goal of maximazing the objective function: F = monotonicity + prognosability.
    
    Parameters:
    - hi_3 (pd.Series): of shape (n,)
    - hi_4 (pd.Series): of shape (n,)
    - vhi_1 (pd.Series): of shape (n,)
    - vhi_2 (pd.Series): of shape (n,)
    
    Returns:
    - higa (pd.Series): of shape (n,)
    '''
    return vhi_1 * (hi_4 - (vhi_2 + 0.5*hi_3) / vhi_2) + 1