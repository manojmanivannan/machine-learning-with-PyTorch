import numpy as np

def create_anomaly(np_arrays,n_points=100,f_low=-2,f_high=2):
    """
    np_array: A numpy array, whose elements need to be modified
    n_points: Number of data points to be modified
    """
    assert(len(np_arrays[0])==len(s) for s in np_arrays)

    def random_non_zero_float(factor_low,factor_high):
        """
        factor_low: the lower limit (default -2)
        factor_high: the upper limit (default 2)
        Returns a random non-zero floating number between low and high
        """
        while True:
            r_float = np.random.uniform(factor_low,factor_high)
            if r_float != 0.0: return r_float
                
    return_arr = []
    # Randomly change some values
    indices_to_change = np.random.choice(len(np_arrays[0]),n_points)
    for arr in np_arrays:
        arr[indices_to_change] = arr[indices_to_change]*random_non_zero_float(f_low,f_high)
        return_arr.append(arr)
    return return_arr


def create_windows(df_numpy, window_size=23, stride=1):
    """
    Function to create windowed sequences
    data: dataframe as numpy array
    windows_size: window size
    stride: steps to move in
    """
    x,y = [] , []
    for i in range(0,len(df_numpy) - window_size,stride):

        x.append(df_numpy[i:i+window_size])  # first 23 data points or records
        y.append(df_numpy[i+window_size])    # next record after the 23rd data point

    return np.array(x), np.array(y)

def z_score_normalize(data, epsilon=1e-10):
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)

    std_non_zero = np.where(std==0,epsilon,std)

    normalized_data = (data - mean)/std_non_zero
    return normalized_data,mean,std

