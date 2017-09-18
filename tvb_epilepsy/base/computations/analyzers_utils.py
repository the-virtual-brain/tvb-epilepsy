import numpy as np
from scipy.signal import butter, lfilter, welch, periodogram, spectrogram
from scipy.interpolate import interp1d, griddata

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
                      nperseg=256, detrend='constant', noverlap=None, f_low=10.0, log_norm=False):
    if freq is None:
        freq = np.linspace(f_low, nperseg, nperseg - f_low - 1)
        df = freq[1] - freq[0]

    psd = []
    for iS in range(x.shape[1]):

        if method is welch:

            f, temp_psd = welch(x[:, iS],
                           fs=fs,  # sample rate
                           nfft=nfft,
                           window=window,   # apply a Hanning window before taking the DFT
                           nperseg=nperseg,        # compute periodograms of 256-long segments of x
                           detrend=detrend,
                           scaling="spectrum",
                           noverlap=noverlap,
                           return_onesided=True,
                           axis=0)
        else:
            f, temp_psd = periodogram(x[:, iS],
                                 fs=fs,  # sample rate
                                 nfft=nfft,
                                 window=window,  # apply a Hanning window before taking the DFT
                                 detrend=detrend,
                                 scaling="spectrum",
                                 return_onesided=True,
                                 axis=0)

        f = interp1d(f, temp_psd)
        temp_psd = f(freq)
        if output == "density":
            temp_psd /= (np.sum(temp_psd) * df)

        psd.append(temp_psd)

    # Stack them to a ndarray
    psd = np.stack(psd, axis=1)

    if output == "energy":
        return np.sum(psd, axis=0)

    else:
        if log_norm:
            psd = np.log(psd)
        return psd, freq


def time_spectral_analysis(x, fs, freq=None, mode="psd", nfft=None, window='hanning', nperseg=256, detrend='constant',
                           noverlap=None, f_low=10.0, calculate_psd=True, log_norm=False):

    # TODO: add a Continuous Wavelet Transform implementation

    if freq is None:
        freq = np.linspace(f_low, nperseg, nperseg - f_low - 1)

    stf = []
    for iS in range(x.shape[1]):
        f, t, temp_s = spectrogram(x[:, iS], fs=fs, nperseg=nperseg, nfft=nfft, window=window, mode=mode,
                                noverlap=noverlap, detrend=detrend, return_onesided=True, scaling='spectrum', axis=0)

        t_mesh, f_mesh = np.meshgrid(t, f, indexing="ij")

        temp_s = griddata((t_mesh.flatten(), f_mesh.flatten()), temp_s.T.flatten(),
                          tuple(np.meshgrid(t, freq, indexing="ij")), method='linear')

        stf.append(temp_s)

    # Stack them to a ndarray
    stf = np.stack(stf, axis=2)
    if log_norm:
        stf = np.log(stf)

    if calculate_psd:
        psd, _ = spectral_analysis(x, fs, freq=freq, method="periodogram", output="spectrum", nfft=nfft, window=window,
                          nperseg=nperseg, detrend=detrend, noverlap=noverlap, log_norm=log_norm)
        return stf, t, freq, psd
    else:
        return stf, t, freq


# Bivariate

def corrcoef(x):
    n, m = x.shape
    return np.corrcoef(x.T)[np.triu_indices(n, 1, m)].flatten()


def covariance(x):
    n, m = x.shape
    return np.cov(x.T)[np.triu_indices(n, 1, m)].flatten()

# TODO: a function to return a matrix of pairwise lags...

# TODO: multivariate, like PCA, ICA, SVD if needed...
