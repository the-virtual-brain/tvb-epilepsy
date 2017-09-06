import numpy as np
from scipy.signal import butter, lfilter, welch, periodogram

# x is assumed to be data (real numbers) arranged along the first dimension of an ndarray
# this factory makes use of the numpy array properties


# Pointwise analyzers:


def center(x, cntr=0.0):
    return (x.T-np.array(cntr)).T


def scale(x, sc=1.0):
    return (x.T / np.array(sc)).T


def mean_center(x):
    return center(x, np.mean(x, axis=0))


def median_center(x):
    return center(x, np.median(x, axis=0))


def zscore(x):
    return std_norm(mean_center(x))


def max_norm(x):
    return scale(x, np.max(x, axis=0))


def maxabs_norm(x):
    return scale(x, np.max(np.abs(x), axis=0))


def std_norm(x):
    return scale(x, np.std(x, axis=0))


def interval_scaling(x, min_targ=0.0, max_targ=1.0, min_orig=None, max_orig=None):
    if min_orig is None:
        min_orig = np.min(x, axis=0)
    if max_orig is None:
        max_orig = np.max(x, axis=0)
    scale_factor = (max_targ - max_orig) / (min_targ - min_orig)
    return max_orig + (x - min_orig) * scale_factor


def threshold(x, th=0.0, out=None):
    if out is None:
        out = th
    x[x < th] = out
    return x


def subthreshold(x, th=0.0, out=None):
    if out is None:
        out = th
    x[x > th] = out
    return x


def sigmoid(x, sc=1.0):
    return 1.0 / (1 + np.exp(-sc * x))


def sigmoidal_scaling(x, cntr=0.0, sc=10.0, min_targ=0.0, max_targ=1.0):
    return min_targ + max_targ * sigmoid(maxabs_norm(center(x, cntr)), sc)


def rectify(x):
    return np.abs(x)


def point_power(x, p=2.0):
    return np.power(x, p)


def log(x, base="natural"):
    if base == 10:
        return np.log10(x)
    elif base == 2:
        return np.log2(x)
    else:
        return np.log(x)

# Across points analyzers:

# Univariate:

# Time domain:


def sum_points(x, ratio=True):
    sum = np.sum(x, axis=0)
    if ratio:
        return sum / x.shape[0]
    else:
        return sum


def energy(x):
    return np.sum(x ** 2, axis=0)


def power(x, n=None):
    if n is None:
        n = np.min(x.shape[0], 1)
    return energy(x) / n


# Frequency domain:

def _butterworth_bandpass(lowcut, highcut, fs, order=3):
    """
    Build a diggital Butterworth filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq  # normalize frequency
    high = highcut / nyq  # normalize frequency
    b, a = butter(order, [low, high], btype='band')
    return b, a


def filter_data(data, lowcut, highcut, fs, order=3):
    # get filter coefficients
    b, a = _butterworth_bandpass(lowcut, highcut, fs, order=order)
    # filter data
    y = lfilter(b, a, data)
    return y


def spectral_analysis(x, fs, freq=None, method="periodogram", output="spectrum", nfft=None, window='hanning',
                           nperseg=512, detrend='constant', noverlap=None):
    if freq is None:
        freq = np.linspace(1, nperseg, nperseg)

    if method is welch:
        f, psd = welch(x,
                       fs=fs,  # sample rate
                       nfft=nfft,
                       window=window,   # apply a Hanning window before taking the DFT
                       nperseg=nperseg,        # compute periodograms of 256-long segments of x
                       detrend=detrend,
                       scaling="spectrum",
                       noverlap=noverlap,
                       axis=0)
    else:
        f, psd = periodogram(x,
                             fs=fs,  # sample rate
                             nfft=nfft,
                             window=window,  # apply a Hanning window before taking the DFT
                             detrend=detrend,
                             scaling="spectrum",
                             axis=0)

    # Fit to desired frequency grid:
    df = freq[1] - freq[0]
    p = np.polyfit(f, psd, psd.shape[0])
    for k in range(psd.shape[1]):
        psd[:, k] = np.polyval(p[:, k], freq)
        if output == "density":
            psd[:, k] /= (np.sum(psd[:, k]) * df)

    if output == "energy":
        return np.sum(psd, axis=0)

    else:
        return psd, freq


# Bivariate

def corrcoef(x):
    n, m = x.shape
    return np.corrcoef(x.T)[np.triu_indices(n, 1, m)].flatten()


def covariance(x):
    n, m = x.shape
    return np.cov(x.T)[np.triu_indices(n, 1, m)].flatten()

# TODO: a function to return a matrix of pairwise lags...

# TODO: multivariate, like PCA, ICA, SVD if needed...
