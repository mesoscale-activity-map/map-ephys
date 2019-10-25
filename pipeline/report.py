import datajoint as dj
import numpy as np
import json
import pathlib
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pipeline import experiment, ephys, psth, tracking, lab
from pipeline.plot import behavior_plot, unit_characteristic_plot, unit_psth
from pipeline import get_schema_name
from pipeline.plot.util import _plot_with_sem


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
        water_res_num, sess_date = get_wr_sessdate(key)
        sess_dir = store_directory / water_res_num / sess_date
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
        fn_prefix = f'{water_res_num}_{sess_date}_'

        fig_dict = {}
        for fig, figname in zip((fig1, fig2), ('behavior_performance', 'jaw_phase_dist')):
            fig_fp = sess_dir / (fn_prefix + figname + '.png')
            fig.tight_layout()
            fig.savefig(fig_fp)
            print(f'Generated {fig_fp}')
            fig_dict[figname] = fig_fp.as_posix()

        plt.close('all')
        self.insert1({**key, **fig_dict})


@schema
class ProbeLevelReport(dj.Computed):
    definition = """
    -> ephys.ProbeInsertion
    ---
    clustering_quality: filepath@report_store
    unit_characteristic: filepath@report_store
    group_psth: filepath@report_store
    group_photostim: filepath@report_store
    """

    @property
    def key_source(self):
        # Only process ProbeInsertion with UnitSelectivity computation fully completed
        ks = (ephys.ProbeInsertion & ephys.UnitStat).proj()
        unit = ks.aggr(ephys.Unit & 'unit_quality != "all"', unit_count='count(*)')
        sel_unit = ks.aggr(psth.UnitSelectivity, sel_unit_count='count(*)')
        return unit * sel_unit & 'unit_count = sel_unit_count'

    def make(self, key):
        water_res_num, sess_date = get_wr_sessdate(key)
        sess_dir = store_directory / water_res_num / sess_date / str(key['insertion_number'])
        sess_dir.mkdir(parents=True, exist_ok=True)

        probe_insertion = ephys.ProbeInsertion & key
        units = ephys.Unit & key

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

        # ---- group_psth ----
        fig3 = unit_characteristic_plot.plot_stacked_contra_ipsi_psth(units)

        # ---- group_photostim ----
        stim_locs = (experiment.Photostim & probe_insertion).fetch('brain_location_name')
        fig4, axs = plt.subplots(1, 1 + len(stim_locs), figsize=(16, 6))
        for pos, stim_loc in enumerate(stim_locs):
            unit_characteristic_plot.plot_psth_photostim_effect(units,
                                                                condition_name_kw=[stim_loc],
                                                                axs=np.array([axs[0], axs[pos + 1]]))
            if pos < len(stim_locs) - 1:
                axs[0].clear()

        [a.set_title(title.replace('_', ' ').upper()) for a, title in zip(axs, ['control'] + list(stim_locs))]

        # ---- Save fig and insert ----
        fn_prefix = f'{water_res_num}_{sess_date}_{key["insertion_number"]}_'

        fig_dict = {}
        for fig, figname in zip((fig1, fig2, fig3, fig4),
                                ('clustering_quality', 'unit_characteristic', 'group_psth', 'group_photostim')):
            fig_fp = sess_dir / (fn_prefix + figname + '.png')
            fig.tight_layout()
            fig.savefig(fig_fp)
            print(f'Generated {fig_fp}')
            fig_dict[figname] = fig_fp.as_posix()

        plt.close('all')
        self.insert1({**key, **fig_dict})


@schema
class UnitLevelReport(dj.Computed):
    definition = """
    -> ephys.Unit
    ---
    unit_psth: filepath@report_store
    unit_behavior: filepath@report_store
    """

    key_source = ephys.Unit & psth.UnitSelectivity

    def make(self, key):
        water_res_num, sess_date = get_wr_sessdate(key)
        sess_dir = store_directory / water_res_num / sess_date / str(key['insertion_number']) / 'units'
        sess_dir.mkdir(parents=True, exist_ok=True)

        fig1 = unit_psth.plot_unit_psth(key)

        fig2 = plt.figure(figsize = (16, 16))
        gs = GridSpec(4, 2)

        # FIXME: configurable 'trial_offset' and 'trial_limit'?
        behavior_plot.plot_tracking(experiment.Session & key, key, tracking_feature='jaw_x', xlim=(-0.5, 1),
                                    trial_offset=10, trial_limit=30, axs=np.array([fig2.add_subplot(gs[:3, col])
                                                                                  for col in range(2)]))

        axs = np.array([fig2.add_subplot(gs[-1, col], polar = True) for col in range(2)])
        behavior_plot.plot_unit_jaw_phase_dist(experiment.Session & key, key, axs=axs)
        [a.set_title('') for a in axs]

        fig2.subplots_adjust(wspace=0.4)
        fig2.subplots_adjust(hspace=0.6)

        # ---- Save fig and insert ----
        fn_prefix = f'{water_res_num}_{sess_date}_{key["insertion_number"]}_u{key["unit"]:03}_'

        fig_dict = {}
        for fig, figname in zip((fig1, fig2), ('unit_psth', 'unit_behavior')):
            fig_fp = sess_dir / (fn_prefix + figname + '.png')
            fig.tight_layout()
            fig.savefig(fig_fp)
            print(f'Generated {fig_fp}')
            fig_dict[figname] = fig_fp.as_posix()

        plt.close('all')
        self.insert1({**key, **fig_dict})


@schema
class SessionLevelCDReport(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    coding_direction: filepath@report_store
    """

    @property
    def key_source(self):
        # Only process Session with UnitSelectivity computation fully completed
        ks = experiment.Session.aggr(ephys.ProbeInsertion, probe_count='count(*)') & 'probe_count > 1'
        unit = ks.aggr(ephys.Unit & 'unit_quality != "all"', unit_count='count(*)')
        sel_unit = ks.aggr(psth.UnitSelectivity, sel_unit_count='count(*)')
        return unit * sel_unit & 'unit_count = sel_unit_count'

    def make(self, key):
        time_period = (-0.4, 0)
        probe_keys = (ephys.ProbeInsertion & key).fetch('KEY')

        for probe in probe_keys:
            units = ephys.Unit & probe
            _, proj_contra_trial, proj_ipsi_trial, time_stamps = psth.compute_CD_projected_psth(
                units.fetch('KEY'), time_period=time_period)

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            _plot_with_sem(proj_contra_trial, time_stamps, ax=ax, c='b')
            _plot_with_sem(proj_ipsi_trial, time_stamps, ax=ax, c='r')

        pass
        # fig1 = unit_characteristic_plot.plot_paired_coding_direction(units & 'brain_area = "thalamus"',
        #                                                              units & 'brain_area = "alm"',
        #                                                              labels = ('thalamus', 'alm'), time_period = (-0.4, 0))


# ---------- HELPER FUNCTIONS --------------
def get_wr_sessdate(key):
    water_res_num, sess_date = (lab.WaterRestriction * experiment.Session & key).fetch1(
        'water_restriction_number', 'session_date')
    return water_res_num, datetime.strftime(sess_date, '%Y%m%d')

