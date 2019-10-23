import datajoint as dj
import numpy as np
import json
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt

from pipeline import experiment, ephys, psth, tracking
from pipeline.plot import behavior_plot, unit_characteristic_plot
from pipeline import get_schema_name

schema = dj.schema(get_schema_name('report'))
mpl.rcParams['font.size'] = 16


store_directory = pathlib.Path(dj.config['stores']['report_store']['location'])


@schema
class SessionLevelReport(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    behavior_performance: filepath@report_store
    jaw_phase_dist: filepath@report_store
    """

    key_source = experiment.Session & experiment.BehaviorTrial & tracking.Tracking

    def make(self, key):
        sess_dir = store_directory / str(key['subject_id']) / str(key['session'])
        sess_dir.mkdir(parents=True, exist_ok=True)

        # ---- behavior_performance ----
        # photostim
        photostims = (experiment.Photostim * experiment.BrainLocation & key).fetch(as_dict=True, order_by='brain_area')

        fig1, axs = plt.subplots(int(1 + np.ceil(len(photostims) / 3)), 3, figsize=(16, 16))
        fig1.subplots_adjust(wspace=0.5)
        [a.axis('off') for a in axs.flatten()]

        gs = axs.flatten()[0].get_gridspec()
        [a.remove() for a in axs.flatten()[:2]]
        ax1 = fig1.add_subplot(gs[0, :])

        # the plot part
        behavior_plot.plot_correct_proportion(key, axs=ax1)
        ax1.axis('on')

        for ax, stim_key in zip(axs.flatten()[3:], photostims):
            behavior_plot.plot_photostim_effect(key, stim_key, axs=ax,
                                                title=stim_key['brain_location_name'].replace('_', ' ').upper())
            ax.axis('on')

        # ---- Jaw movement phase distribution ----
        fig2 = behavior_plot.plot_windowed_jaw_phase_dist(key, xlim=(-0.12, 0.3), w_size=0.01, bin_counts=20)

        # ---- Save fig and insert ----
        fig1_fp = sess_dir / 'behavior_performance.png'
        fig2_fp = sess_dir / 'jaw_phase_dist.png'

        fig1.savefig(fig1_fp)
        print(f'Generated {fig1_fp}')
        fig2.savefig(fig2_fp)
        print(f'Generated {fig2_fp}')
        self.insert1(dict(key, behavior_performance=fig1_fp.as_posix(),
                          jaw_phase_dist=fig2_fp.as_posix()))


@schema
class ProbeLevelReport(dj.Computed):
    definition = """
    -> ephys.ProbeInsertion
    ---
    clustering_quality: filepath@report_store
    unit_characteristic: filepath@report_store
    """

    key_source = ephys.ProbeInsertion & ephys.UnitStat & psth.UnitSelectivity

    def make(self, key):
        sess_dir = store_directory / str(key['subject_id']) / str(key['session']) / str(key['insertion_number'])
        sess_dir.mkdir(parents=True, exist_ok=True)

        probe_insertion = ephys.ProbeInsertion & key

        # ---- clustering_quality ----
        fig1, axs = plt.subplots(4, 3, figsize=(16, 16))
        fig1.subplots_adjust(wspace=0.5)
        fig1.subplots_adjust(hspace=0.5)

        gs = axs[0, 0].get_gridspec()
        [a.remove() for a in axs[2:, :].flatten()]

        unit_characteristic_plot.plot_clustering_quality(probe_insertion, axs=axs[:2, :])
        unit_characteristic_plot.plot_unit_characteristic(probe_insertion, axs=np.array([fig1.add_subplot(gs[2:, col])
                                                                                         for col in range(3)]))

        # ---- unit_characteristic ----
        fig2, axs = plt.subplots(1, 4, figsize=(16, 16))
        fig2.subplots_adjust(wspace=0.8)

        unit_characteristic_plot.plot_unit_selectivity(probe_insertion, axs=axs[:3])
        unit_characteristic_plot.plot_unit_bilateral_photostim_effect(probe_insertion, axs=axs[-1])

        # ---- Save fig and insert ----
        fig1_fp = sess_dir / 'clustering_quality.png'
        fig2_fp = sess_dir / 'unit_characteristic.png'

        fig1.savefig(fig1_fp)
        print(f'Generated {fig1_fp}')
        fig2.savefig(fig2_fp)
        print(f'Generated {fig2_fp}')
        self.insert1(dict(key, clustering_quality=fig1_fp.as_posix(),
                          unit_characteristic=fig2_fp.as_posix()))


@schema
class UnitLevelReport(dj.Computed):
    definition = """
    -> ephys.ProbeInsertion
    ---
    clustering_quality: filepath@report_store
    unit_characteristic: filepath@report_store
    """

    key_source = ephys.Unit & psth.UnitSelectivity

    def make(self, key):
        pass


