
import numpy as np
from scipy.signal import decimate

from tvb_epilepsy.base.computations.analyzers_utils import filter_data


def decimate_signals(time, signals, decim_ratio):
    signals = decimate(signals, decim_ratio, axis=0, zero_phase=True)
    time = decimate(time, decim_ratio, zero_phase=True)
    dt = np.mean(time)
    (n_times, n_signals) = signals.shape
    return signals, time, dt, n_times


def cut_signals_tails(time, signals, cut_tails):
    signals = signals[cut_tails[0]:-cut_tails[-1]]
    time = time[cut_tails[0]:-cut_tails[-1]]
    (n_times, n_signals) = signals.shape
    return signals, time, n_times


# def compute_envelope(data, time, samp_rate, hp_freq=5.0, lp_freq=0.1, benv_cut=100, cut_tails=None, order=3):
#     if cut_tails is not None:
#         start = cut_tails[0]
#         stop = cut_tails[1]
#     else:
#         start = int(samp_rate / lp_freq)
#         skip = int(samp_rate / (lp_freq * 3))
#     data = filter_data(data, samp_rate, 'highpass', lowcut=lp_freq, order=order)
#     data = filter_data(np.abs(data), samp_rate, 'lowpass', highcut=hp_freq, order=order)
#     data = data[:, start::skip]
#     fm = benv > 100  # bipolar 100, otherwise 300 (roughly)
#     incl_names = "HH1-2 HH2-3".split()
#     incl_idx = np.array([i for i, (name, *_) in enumerate(contacts_bip) if name in incl_names])
#     incl = np.setxor1d(
#         np.unique(np.r_[
#                       incl_idx,
#                       np.r_[:len(fm)][fm.any(axis=1)]
#                   ])
#         , afc_idx)
#     isort = incl[np.argsort([te[fm[i]].mean() for i in incl])]
#     iother = np.setxor1d(np.r_[:len(benv)], isort)
#     lbenv = np.log(np.clip(benv[isort], benv[benv > 0].min(), None))
#     lbenv_all = np.log(np.clip(benv, benv[benv > 0].min(), None))
#     return te, isort, iother, lbenv, lbenv_all