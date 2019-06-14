import numpy as np
import scipy as sp
import datajoint as dj

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd

from scipy import signal

from pipeline import experiment, tracking, ephys, psth


def plot_clustering_quality(session_key):
    amp, snr, spk_times = (ephys.Unit * ephys.ProbeInsertion.InsertionLocation & session_key).fetch(
        'unit_amp', 'unit_snr', 'spike_times')
    isi_violation, spk_rate = zip(*((compute_isi_violation(spk), compute_spike_rate(spk)) for spk in spk_times))

    metrics = {'amp': amp,
               'snr': snr,
               'isi': np.array(isi_violation),
               'rate': np.array(spk_rate)}
    label_mapper = {'amp': 'Amplitude',
                    'snr': 'Signal to noise ration (SNR)',
                    'isi': 'ISI violation (%)',
                    'rate': 'Firing rate (spike/s)'}

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.subplots_adjust(wspace=0.4)

    for (m1, m2), ax in zip(itertools.combinations(list(metrics.keys()), 2), axs.flatten()):
        ax.plot(metrics[m1], metrics[m2], '.k')
        ax.set_xlabel(label_mapper[m1])
        ax.set_ylabel(label_mapper[m2])

        # cosmetic
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


def plot_unit_characteristic(session_key):
    amp, snr, spk_times, x, y, insertion_depth = (ephys.Unit * ephys.ProbeInsertion.InsertionLocation
                                                  & session_key & 'unit_quality = "good"').fetch(
        'unit_amp', 'unit_snr', 'spike_times', 'unit_posx', 'unit_posy', 'dv_location')

    spk_rate = np.array(list(compute_spike_rate(spk) for spk in spk_times))
    insertion_depth = np.where(np.isnan(insertion_depth), 0, insertion_depth)

    metrics = pd.DataFrame(list(zip(*(amp/amp.max(), snr/snr.max(), spk_rate/spk_rate.max(), x, y + insertion_depth))))
    metrics.columns = ['amp', 'snr', 'rate', 'x', 'y']

    fig, axs = plt.subplots(1, 3, figsize=(10, 8))
    fig.subplots_adjust(wspace=0.6)

    cosmetic = {'legend': None,
                'linewidth': 1.75,
                'alpha': 0.9,
                'facecolor': 'none', 'edgecolor': 'k'}
    m_scale = 1200

    sns.scatterplot(data=metrics, x='x', y='y', s=metrics.amp*m_scale, ax=axs[0], **cosmetic)
    sns.scatterplot(data=metrics, x='x', y='y', s=metrics.snr*m_scale, ax=axs[1], **cosmetic)
    sns.scatterplot(data=metrics, x='x', y='y', s=metrics.rate*m_scale, ax=axs[2], **cosmetic)

    # cosmetic
    for title, ax in zip(('Amplitude', 'SNR', 'Firing rate'), axs.flatten()):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(title)
        ax.set_xlim((-10, 60))


def plot_unit_selectivity(session_key):
    contra_selective_units = (psth.UnitSelectivity * ephys.Unit & session_key
                              & 'unit_quality = "good"' - 'selectivity = "contra-selective"')


def compute_isi_violation(spike_times, isi_thresh=2):
    isi = np.diff(spike_times)
    return sum((isi < isi_thresh).astype(int)) / len(isi)


def compute_spike_rate(spike_times):
    return len(spike_times) #/ (spike_times[-1] - spike_times[0])
