import numpy as np


def ED_distance(ts1, ts2):
    """
    Calculate the Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    ed_dist : float
        Euclidean distance between ts1 and ts2.
    """

    if len(ts1) != len(ts2):
        raise ValueError("The two arrays must have the same length.")

    ed_dist = np.sqrt(np.sum((ts1 - ts2)**2))

    return ed_dist


def norm_ED_distance(ts1, ts2):
    """
    Calculate the normalized Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    norm_ed_dist : float
        The normalized Euclidean distance between ts1 and ts2.
    """

    if len(ts1) != len(ts2):
        raise ValueError("The two arrays must have the same length.")

    m = len(ts1)

    norm_ed_dist = np.sqrt(abs(2*m * (1 - (np.dot(ts1, ts2) - m * (sum(ts1)/m) * (sum(ts2)/m)) / (
        m * np.sqrt(sum(ts1**2 - (sum(ts1)/m)**2) / m) * np.sqrt(sum(ts2**2 - (sum(ts2)/m)**2) / m)))))

    return norm_ed_dist


def DTW_distance(ts1, ts2, r=1.0):
    """
    Calculate DTW distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    r : float
        Warping window size.

    Returns
    -------
    dtw_dist : float
        DTW distance between ts1 and ts2.
    """

    n = len(ts1)
    m = len(ts2)

    # Матрица расстояний
    dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix[:, :] = np.inf
    dtw_matrix[0, 0] = 0

    # Вычисление DTW меры
    for i in range(1, n+1):
        for j in range(max(1, i-int(np.floor(m*r))), min(m, i+int(np.floor(m*r))) + 1):
            cost = np.square(ts1[i-1] - ts2[j-1])
            dtw_matrix[i, j] = cost + \
                min(dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1])

    dtw_dist = dtw_matrix[n, m]

    return dtw_dist


def calculate_distance_matrix(data, metric='euclidean', normalize=True):

    N = data.shape[0] # number of time series
    
    if metric=='euclidean':
        if normalize:
            dist_func = norm_ED_distance
        else:
            dist_func = ED_distance
    elif metric=='dtw':
        if normalize:
            for i in range(N):
                data[i] = z_normalize(data[i])
        dist_func = DTW_distance
    else:
        raise ValueError("Metric must be 'euclidean' or 'dtw'.")

    # Initialize the distance matrix
    distance_matrix = np.zeros(shape=(N, N))
    
    for i in range(N):
        for j in range(i, N):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                distance_matrix[i, j] = dist_func(data[i], data[j])
    
    distance_matrix = distance_matrix + distance_matrix.T

    return distance_matrix