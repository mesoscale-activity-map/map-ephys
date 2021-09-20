import os
import datajoint as dj
import numpy as np
import pathlib
from datetime import datetime
import pandas as pd
import uuid

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import io
from PIL import Image
import itertools

from pipeline import get_schema_name, FailedUnitCriteriaError
from pipeline import (experiment, ephys, psth, tracking, lab, histology, ccf,
                      foraging_analysis, oralfacial_analysis)

from pipeline.plot import behavior_plot, unit_characteristic_plot, unit_psth, histology_plot, PhotostimError, foraging_plot
from pipeline.plot.util import _plot_with_sem, _jointplot_w_hue
from pipeline.util import _get_trial_event_times
from pipeline.mtl_analysis import helper_functions as mtl_plot

import warnings
warnings.filterwarnings('ignore')


schema = dj.schema(get_schema_name('report'))

os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = "TRUE"

DEFAULT_REPORT_STORE = {
    "protocol": "s3",
    "endpoint": "s3.amazonaws.com",
    "bucket": "map-report",
    "location": "report/v2",
    "stage": "./data/report_stage",
    "access_key": "",
    "secret_key": ""
}

if 'stores' not in dj.config:
    dj.config['stores'] = {}

if 'report_store' not in dj.config['stores']:
    dj.config['stores']['report_store'] = DEFAULT_REPORT_STORE

report_cfg = dj.config['stores']['report_store']

if report_cfg['protocol'] == 's3':
    store_location = (pathlib.Path(report_cfg['bucket'])
                      / pathlib.Path(report_cfg['location']))
    store_location = 'S3: ' + str(store_location)
else:
    store_location = pathlib.Path(report_cfg['location'])


store_stage = pathlib.Path(report_cfg['stage'])

mpl.rcParams['font.size'] = 16


# ============================= SESSION LEVEL ====================================


@schema
class SessionLevelReport(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    behavior_performance: filepath@report_store
    """

    key_source = experiment.Session & experiment.BehaviorTrial & experiment.PhotostimBrainRegion

    def make(self, key):
        water_res_num, sess_date = get_wr_sessdate(key)
        sess_dir = store_stage / water_res_num / sess_date
        sess_dir.mkdir(parents=True, exist_ok=True)

        # ---- behavior_performance ----
        # photostim
        photostims = (experiment.Photostim * experiment.PhotostimBrainRegion & key).fetch(as_dict=True,
                                                                                          order_by='stim_brain_area')

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
            stim_loc = ' '.join([stim_key['stim_laterality'], stim_key['stim_brain_area']]).upper()
            try:
                behavior_plot.plot_photostim_effect(key, stim_key, axs=ax, title=stim_loc)
            except ValueError:
                ax.remove()
            ax.axis('on')

        # ---- Save fig and insert ----
        fn_prefix = f'{water_res_num}_{sess_date}_'
        fig_dict = save_figs((fig1,), ('behavior_performance',), sess_dir, fn_prefix)

        plt.close('all')
        self.insert1({**key, **fig_dict})


@schema
class SessionLevelCDReport(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    cd_probe_count: int
    coding_direction: filepath@report_store
    """

    @property
    def key_source(self):
        # Only process Session with UnitSelectivity computation fully completed
        # - only on probe insertions with RecordableBrainRegion
        ks = experiment.Session.aggr(ephys.ProbeInsertion, probe_count='count(*)')
        ks = ks - (ephys.ProbeInsertion - ephys.ProbeInsertion.RecordableBrainRegion)
        unit = ks.aggr(ephys.Unit & 'unit_quality != "all"', unit_count='count(*)')
        sel_unit = ks.aggr(psth.UnitSelectivity, sel_unit_count='count(*)')
        return unit * sel_unit & 'unit_count = sel_unit_count'

    def make(self, key):
        water_res_num, sess_date = get_wr_sessdate(key)
        sess_dir = store_stage / water_res_num / sess_date
        sess_dir.mkdir(parents=True, exist_ok=True)

        # ---- Setup ----
        time_period = (-0.4, 0)
        probe_keys = (ephys.ProbeInsertion & key).fetch('KEY', order_by='insertion_number')

        fig1, axs = plt.subplots(len(probe_keys), len(probe_keys), figsize=(16, 16))

        if len(probe_keys) > 1:
            [a.axis('off') for a in axs.flatten()]

            # ---- Plot Coding Direction per probe ----
            probe_proj = {}
            for pid, probe in enumerate(probe_keys):
                units = ephys.Unit & probe
                label = (ephys.ProbeInsertion & probe).aggr(ephys.ProbeInsertion.RecordableBrainRegion.proj(
                    brain_region='CONCAT(hemisphere, " ", brain_area)'),
                    brain_regions='GROUP_CONCAT(brain_region SEPARATOR", ")').fetch1('brain_regions')
                label = '({}) {}'.format(probe['insertion_number'], label)

                _, period_starts = _get_trial_event_times(['sample', 'delay', 'go'], units, 'good_noearlylick_hit')

                # ---- compute CD projected PSTH ----
                _, proj_contra_trial, proj_ipsi_trial, time_stamps, hemi = psth.compute_CD_projected_psth(
                    units.fetch('KEY'), time_period=time_period)

                # ---- save projection results ----
                probe_proj[pid] = (proj_contra_trial, proj_ipsi_trial, time_stamps, label, hemi)

                # ---- generate fig with CD plot for this probe ----
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                _plot_with_sem(proj_contra_trial, time_stamps, ax=ax, c='b')
                _plot_with_sem(proj_ipsi_trial, time_stamps, ax=ax, c='r')
                # cosmetic
                for x in period_starts:
                    ax.axvline(x=x, linestyle = '--', color = 'k')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_ylabel('CD projection (a.u.)')
                ax.set_xlabel('Time (s)')
                ax.set_title(label)
                fig.tight_layout()

                # ---- plot this fig into the main figure ----
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                axs[pid, pid].imshow(Image.open(buf))
                buf.close()
                plt.close(fig)

            # ---- Plot probe-pair correlation ----
            for p1, p2 in itertools.combinations(probe_proj.keys(), r=2):
                proj_contra_trial_g1, proj_ipsi_trial_g1, time_stamps, label_g1, p1_hemi = probe_proj[p1]
                proj_contra_trial_g2, proj_ipsi_trial_g2, time_stamps, label_g2, p2_hemi = probe_proj[p2]
                labels = [label_g1, label_g2]

                # plot trial CD-endpoint correlation
                p_start, p_end = time_period
                contra_cdend_1 = proj_contra_trial_g1[:, np.logical_and(
                    time_stamps >= p_start, time_stamps < p_end)].mean(axis=1)
                ipsi_cdend_1 = proj_ipsi_trial_g1[:, np.logical_and(
                    time_stamps >= p_start, time_stamps < p_end)].mean(axis=1)
                if p1_hemi == p2_hemi:
                    contra_cdend_2 = proj_contra_trial_g2[:, np.logical_and(
                        time_stamps >= p_start, time_stamps < p_end)].mean(axis=1)
                    ipsi_cdend_2 = proj_ipsi_trial_g2[:, np.logical_and(
                        time_stamps >= p_start, time_stamps < p_end)].mean(axis=1)
                else:
                    contra_cdend_2 = proj_ipsi_trial_g2[:, np.logical_and(
                        time_stamps >= p_start, time_stamps < p_end)].mean(axis=1)
                    ipsi_cdend_2 = proj_contra_trial_g2[:, np.logical_and(
                        time_stamps >= p_start, time_stamps < p_end)].mean(axis=1)

                c_df = pd.DataFrame([contra_cdend_1, contra_cdend_2]).T
                c_df.columns = labels
                c_df['trial-type'] = 'contra'
                i_df = pd.DataFrame([ipsi_cdend_1, ipsi_cdend_2]).T
                i_df.columns = labels
                i_df['trial-type'] = 'ipsi'
                df = c_df.append(i_df)

                # remove NaN trial - could be due to some trials having no spikes
                non_nan = ~np.logical_or(np.isnan(df[labels[0]]).values, np.isnan(df[labels[1]]).values)
                df = df[non_nan]

                fig = plt.figure(figsize=(6, 6))
                jplot = _jointplot_w_hue(data=df, x=labels[0], y=labels[1], hue= 'trial-type', colormap=['b', 'r'],
                                         figsize=(8, 6), fig=fig, scatter_kws=None)

                # ---- plot this fig into the main figure ----
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                axs[p1, p2].imshow(Image.open(buf))
                buf.close()
                plt.close(fig)

        else:
            # ---- Plot Single-Probe Coding Direction ----
            probe = probe_keys[0]
            units = ephys.Unit & probe
            label = (ephys.ProbeInsertion & probe).aggr(ephys.ProbeInsertion.RecordableBrainRegion.proj(
                brain_region = 'CONCAT(hemisphere, " ", brain_area)'),
                brain_regions = 'GROUP_CONCAT(brain_region SEPARATOR", ")').fetch1('brain_regions')

            unit_characteristic_plot.plot_coding_direction(units, time_period=time_period, label=label, axs=axs)

        # ---- Save fig and insert ----
        fn_prefix = f'{water_res_num}_{sess_date}_'
        fig_dict = save_figs((fig1,), ('coding_direction',), sess_dir, fn_prefix)

        plt.close('all')
        self.insert1({**key, **fig_dict, 'cd_probe_count': len(probe_keys)})


@schema
class SessionLevelProbeTrack(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    session_tracks_plot: filepath@report_store
    probe_track_count: int
    probe_tracks: longblob
    """

    key_source = experiment.Session & histology.LabeledProbeTrack

    def make(self, key):
        water_res_num, sess_date = get_wr_sessdate(key)
        sess_dir = store_stage / water_res_num / sess_date
        sess_dir.mkdir(parents=True, exist_ok=True)

        fig1 = plt.figure(figsize=(16, 12))

        for axloc, elev, azim in zip((221, 222, 223, 224), (65, 0, 90, 0), (-15, 0, 0, 90)):
            ax = fig1.add_subplot(axloc, projection='3d')
            ax.view_init(elev, azim)
            probe_tracks = histology_plot.plot_probe_tracks(key, ax=ax)

        # ---- Save fig and insert ----
        fn_prefix = f'{water_res_num}_{sess_date}_'
        fig_dict = save_figs((fig1,), ('session_tracks_plot',), sess_dir, fn_prefix)

        plt.close('all')
        self.insert1({**key, **fig_dict, 'probe_track_count': len(probe_tracks), 'probe_tracks': probe_tracks})


@schema
class SessionLevelForagingSummary(dj.Computed):
    definition = """
    # Marton's foraging inspector with default settings
    -> experiment.Session
    ---
    session_foraging_summary: filepath@report_store
    """
    
    # Only 2-lp plots
    key_source = experiment.Session & (experiment.BehaviorTrial & 'task = "foraging"')
    
    # Default settings for plots
    choice_averaging_window = 10
    choice_kernel = np.ones(choice_averaging_window) / choice_averaging_window
    # invoke plot functions   
    filters = {'ignore_rate_max': 100,
               'averaging_window': choice_averaging_window}
    local_matching = {'calculate_local_matching': True,
                      'sliding_window': 100,
                      'matching_window': 300,
                      'matching_step': 30,
                      'efficiency_type': 'ideal'}
    
    def make(self, key):
        water_res_num, sess_date = get_wr_sessdate(key)
        sess_dir = store_stage / water_res_num / sess_date
        sess_dir.mkdir(parents=True, exist_ok=True)
        
        # --- the plot part ---
        fig = plt.figure(figsize=(15,20))
        gs = GridSpec(9, 1, wspace=0.4, hspace=0.4, bottom=0.07, top=0.95, left=0.1, right=0.9) 
        ax1 = fig.add_subplot(gs[0:2, :])
        ax1.set_title(f'{water_res_num}, session {key["session"]}')
        ax2 = fig.add_subplot(gs[2, :])
        ax3 = fig.add_subplot(gs[3:5, :])
        ax4 = fig.add_subplot(gs[5:7, :])
        ax5 = fig.add_subplot(gs[7:9, :])
        ax1.get_shared_x_axes().join(ax1, ax2, ax3, ax4, ax5)

        df_behaviortrial = foraging_plot.extract_trials(plottype = '2lickport',
                                          wr_name = water_res_num,
                                          sessions = [key['session'], key['session']],
                                          show_bias_check_trials = True,
                                          kernel = self.choice_kernel,
                                          filters = self.filters,
                                          local_matching = self.local_matching)
        
        foraging_plot.plot_trials(df_behaviortrial,
                    ax1,
                    ax2,
                    plottype = '2lickport',
                    wr_name = water_res_num,
                    sessions = [key['session'], key['session']],
                    plot_every_choice = True,
                    show_bias_check_trials = True,
                    choice_filter = self.choice_kernel)
        
        foraging_plot.plot_local_efficiency_matching_bias(df_behaviortrial,
                                            ax3)
        foraging_plot.plot_rt_iti(df_behaviortrial,
                    ax4,
                    ax5,
                    plottype = '2lickport',
                    wr_name = water_res_num,
                    sessions = [key['session'], key['session']],
                    show_bias_check_trials = True,
                    kernel = self.choice_kernel
                    )
        
        # ---- Save fig and insert ----
        fn_prefix = f'{water_res_num}_{sess_date}_'
        fig_dict = save_figs((fig,), ('session_foraging_summary',), sess_dir, fn_prefix)
        plt.close('all')
        self.insert1({**key, **fig_dict})
        
        
@schema
class SessionLevelForagingLickingPSTH(dj.Computed):
    definition = """
    # Licking events aligned with the GO cue
    -> experiment.Session
    ---
    session_foraging_licking_psth: filepath@report_store
    """
    
    # Only 2-lp plots
    key_source = experiment.Session & (experiment.BehaviorTrial & 'task = "foraging"')
        
    def make(self, key):
        water_res_num, sess_date = get_wr_sessdate(key)
        sess_dir = store_stage / water_res_num / sess_date
        sess_dir.mkdir(parents=True, exist_ok=True)

        # -- Plotting --
        fig = plt.figure(figsize=(20,12))
        fig.suptitle(f'{water_res_num}, session {key["session"]}')
        gs = GridSpec(5, 1, wspace=0.4, hspace=0.4, bottom=0.07, top=0.95, left=0.1, right=0.9) 
        
        ax1 = fig.add_subplot(gs[0:3, :])
        ax2 = fig.add_subplot(gs[3, :])
        ax3 = fig.add_subplot(gs[4, :])
        ax1.get_shared_x_axes().join(ax1, ax2, ax3)
        
        # Plot settings
        plot_setting = {'left lick':'red', 'right lick':'blue'}  
        
        # -- Get event times --
        key_subject_id_session = (experiment.Session() & (lab.WaterRestriction() & 'water_restriction_number="{}"'.format(water_res_num)) 
                                  & 'session="{}"'.format(key['session'])).fetch1("KEY")
        go_cue_times = (experiment.TrialEvent() & key_subject_id_session & 'trial_event_type="go"').fetch('trial_event_time', order_by='trial').astype(float)
        lick_times = pd.DataFrame((experiment.ActionEvent() & key_subject_id_session).fetch(order_by='trial'))
        
        trial_num = len(go_cue_times)
        all_trial_num = np.arange(1, trial_num+1).tolist()
        all_trial_start = [[-x] for x in go_cue_times]
        all_lick = dict()
        for event_type in plot_setting:
            all_lick[event_type] = []
            for i, trial_start in enumerate(all_trial_start):
                all_lick[event_type].append((lick_times[(lick_times['trial']==i+1) & (lick_times['action_event_type']==event_type)]['action_event_time'].values.astype(float) + trial_start).tolist())
        
        # -- All licking events (Ordered by trials) --
        ax1.plot([0, 0], [0, trial_num], 'k', lw=0.5)   # Aligned by go cue
        ax1.set(ylabel='Trial number', xlim=(-3, 3), xticks=[])
    
        # Batch plotting to speed up    
        ax1.eventplot(lineoffsets=all_trial_num, positions=all_trial_start, color='k')   # Aligned by go cue
        for event_type in plot_setting:
            ax1.eventplot(lineoffsets=all_trial_num, 
                         positions=all_lick[event_type],
                         color=plot_setting[event_type],
                         linewidth=2)   # Trial start
        
        # -- Histogram of all licks --
        for event_type in plot_setting:
            sns.histplot(np.hstack(all_lick[event_type]), binwidth=0.01, alpha=0.5, 
                         ax=ax2, color=plot_setting[event_type], label=event_type)  # 10-ms window
        
        ymax_tmp = max(ax2.get_ylim())  
        sns.histplot(-go_cue_times, binwidth=0.01, color='k', ax=ax2, label='trial start')  # 10-ms window
        ax2.axvline(x=0, color='k', lw=0.5)
        ax2.set(ylim=(0, ymax_tmp), xticks=[], title='All events') # Fix the ylim of left and right licks
        ax2.legend()
        
        # -- Histogram of reaction time (first lick after go cue) --   
        plot_setting = {'LEFT':'red', 'RIGHT':'blue'}  
        for water_port in plot_setting:
            this_RT = (foraging_analysis.TrialStats() & key_subject_id_session & (experiment.WaterPortChoice() & 'water_port="{}"'.format(water_port))).fetch('reaction_time').astype(float)
            sns.histplot(this_RT, binwidth=0.01, alpha=0.5, 
                         ax=ax3, color=plot_setting[water_port], label=water_port)  # 10-ms window
        ax3.axvline(x=0, color='k', lw=0.5)
        ax3.set(xlabel='Time to Go Cue (s)', title='First lick (reaction time)') # Fix the ylim of left and right licks
        ax3.legend()
        
        # ---- Save fig and insert ----
        fn_prefix = f'{water_res_num}_{sess_date}_'
        fig_dict = save_figs((fig,), ('session_foraging_licking_psth',), sess_dir, fn_prefix)
        plt.close('all')
        self.insert1({**key, **fig_dict})
        
        
# ============================= PROBE LEVEL ====================================


@schema
class ProbeLevelReport(dj.Computed):
    definition = """
    -> ephys.ProbeInsertion
    -> ephys.ClusteringMethod
    ---
    clustering_quality: filepath@report_store
    unit_characteristic: filepath@report_store
    group_psth: filepath@report_store
    """

    @property
    def key_source(self):
        # Only process ProbeInsertion with UnitSelectivity computation fully completed
        ks = (ephys.ProbeInsertion * ephys.ClusteringMethod & ephys.UnitStat).proj()
        unit = ks.aggr(ephys.Unit & 'unit_quality != "all"', unit_count='count(*)')
        sel_unit = ks.aggr(psth.UnitSelectivity, sel_unit_count='count(*)')
        return unit * sel_unit & 'unit_count = sel_unit_count'

    def make(self, key):
        water_res_num, sess_date = get_wr_sessdate(key)
        probe_dir = store_stage / water_res_num / sess_date / str(key['insertion_number'])
        probe_dir.mkdir(parents=True, exist_ok=True)

        probe_insertion = ephys.ProbeInsertion & key
        units = ephys.Unit & key

        # ---- clustering_quality ----
        fig1, axs = plt.subplots(4, 3, figsize=(16, 16))
        fig1.subplots_adjust(wspace=0.5)
        fig1.subplots_adjust(hspace=0.5)

        gs = axs[0, 0].get_gridspec()
        [a.remove() for a in axs[2:, :].flatten()]

        unit_characteristic_plot.plot_clustering_quality(
            probe_insertion, clustering_method=key['clustering_method'], axs=axs[:2, :])
        unit_characteristic_plot.plot_unit_characteristic(
            probe_insertion, clustering_method=key['clustering_method'], axs=np.array([fig1.add_subplot(gs[2:, col])
                                                                                       for col in range(3)]))

        # ---- unit_characteristic ----
        fig2, axs = plt.subplots(1, 4, figsize=(16, 16))
        fig2.subplots_adjust(wspace=0.8)

        unit_characteristic_plot.plot_unit_selectivity(
            probe_insertion, clustering_method=key['clustering_method'], axs=axs[:3])

        # if photostim performed in this session
        try:
            unit_characteristic_plot.plot_unit_bilateral_photostim_effect(
                probe_insertion, clustering_method=key['clustering_method'], axs=axs[-1])
        except PhotostimError:
            axs[-1].remove()

        # ---- group_psth ----
        fig3 = unit_characteristic_plot.plot_stacked_contra_ipsi_psth(units)

        # ---- Save fig and insert ----
        fn_prefix = f'{water_res_num}_{sess_date}_{key["insertion_number"]}_{key["clustering_method"]}_'
        fig_dict = save_figs((fig1, fig2, fig3), ('clustering_quality', 'unit_characteristic', 'group_psth'),
                             probe_dir, fn_prefix)

        plt.close('all')
        self.insert1({**key, **fig_dict})


@schema
class ProbeLevelPhotostimEffectReport(dj.Computed):
    definition = """
    -> ephys.ProbeInsertion
    -> ephys.ClusteringMethod
    ---
    group_photostim: filepath@report_store
    """

    @property
    def key_source(self):
        # Only process ProbeInsertion with UnitPSTH computation (for all TrialCondition) fully completed
        ks = ephys.ProbeInsertion * ephys.ClusteringMethod & ephys.ProbeInsertion.InsertionLocation
        probe_current_psth = ks.aggr(psth.UnitPsth, present_u_psth_count='count(*)')
        # Note: keep this 'probe_full_psth' query in sync with psth.UnitPSTH.key_source
        probe_full_psth = (ks.aggr(
            (ephys.Unit & 'unit_quality != "all"').proj(), unit_count=f'count(*)') * dj.U().aggr(
            psth.TrialCondition, trial_cond_count='count(*)')).proj(
            full_u_psth_count='unit_count * trial_cond_count')

        return probe_current_psth * probe_full_psth & 'present_u_psth_count = full_u_psth_count'

    def make(self, key):
        water_res_num, sess_date = get_wr_sessdate(key)
        probe_dir = store_stage / water_res_num / sess_date / str(key['insertion_number'])
        probe_dir.mkdir(parents=True, exist_ok=True)

        probe_insertion = ephys.ProbeInsertion & key
        units = ephys.Unit & key

        # ---- group_photostim ----
        stim_locs = (experiment.PhotostimBrainRegion & probe_insertion).proj(
            brain_region='CONCAT(stim_laterality, "_", stim_brain_area)').fetch('brain_region')

        axs_to_be_removed = []
        fig1, axs = plt.subplots(1, 1 + len(stim_locs), figsize=(16, 6))
        for pos, stim_loc in enumerate(stim_locs):
            try:
                unit_characteristic_plot.plot_psth_photostim_effect(units,
                                                                    condition_name_kw=[stim_loc.lower()],
                                                                    axs=np.array([axs[0], axs[pos + 1]]))
            except ValueError:
                axs_to_be_removed.append(axs[pos + 1])

            if pos < len(stim_locs) - 1:
                axs[0].clear()

        [a.set_title(title.replace('_', ' ').upper()) for a, title in zip(axs, ['control'] + list(stim_locs))]
        [a.remove() for a in axs_to_be_removed]

        # ---- Save fig and insert ----
        fn_prefix = f'{water_res_num}_{sess_date}_{key["insertion_number"]}_{key["clustering_method"]}_'
        fig_dict = save_figs((fig1,), ('group_photostim',), probe_dir, fn_prefix)

        plt.close('all')
        self.insert1({**key, **fig_dict})


@schema
class ProbeLevelDriftMap(dj.Computed):
    definition = """
    -> ephys.ProbeInsertion
    -> ephys.ClusteringMethod
    shank: int
    ---
    driftmap: filepath@report_store
    """

    # Only process ProbeInsertion with Histology and InsertionLocation known
    key_source = (ephys.ProbeInsertion * ephys.ClusteringMethod & ephys.Unit.proj()) & histology.InterpolatedShankTrack

    def make(self, key):
        water_res_num, sess_date = get_wr_sessdate(key)
        probe_dir = store_stage / water_res_num / sess_date / str(key['insertion_number'])
        probe_dir.mkdir(parents=True, exist_ok=True)

        probe_insertion = ephys.ProbeInsertion & key

        shanks = probe_insertion.aggr(lab.ElectrodeConfig.Electrode * lab.ProbeType.Electrode,
                                      shanks='GROUP_CONCAT(DISTINCT shank SEPARATOR ", ")').fetch1('shanks')
        shanks = np.array(shanks.split(', ')).astype(int)

        for shank in shanks:
            fig = unit_characteristic_plot.plot_driftmap(probe_insertion, shank_no=shank)
            # ---- Save fig and insert ----
            fn_prefix = f'{water_res_num}_{sess_date}_{key["insertion_number"]}_{key["clustering_method"]}_{shank}_'
            fig_dict = save_figs((fig,), ('driftmap',), probe_dir, fn_prefix)
            plt.close('all')
            self.insert1({**key, **fig_dict, 'shank': shank})


@schema
class ProbeLevelCoronalSlice(dj.Computed):
    definition = """
    -> ephys.ProbeInsertion
    shank: int
    ---
    coronal_slice: filepath@report_store
    """

    # Only process ProbeInsertion with Histology and ElectrodeCCFPosition known
    key_source = ephys.ProbeInsertion & histology.ElectrodeCCFPosition

    def make(self, key):
        water_res_num, sess_date = get_wr_sessdate(key)
        probe_dir = store_stage / water_res_num / sess_date / str(key['insertion_number'])
        probe_dir.mkdir(parents=True, exist_ok=True)

        probe_insertion = ephys.ProbeInsertion & key

        shanks = probe_insertion.aggr(lab.ElectrodeConfig.Electrode * lab.ProbeType.Electrode,
                                      shanks='GROUP_CONCAT(DISTINCT shank SEPARATOR ", ")').fetch1('shanks')
        shanks = np.array(shanks.split(', ')).astype(int)

        for shank in shanks:
            fig = unit_characteristic_plot.plot_pseudocoronal_slice(probe_insertion, shank_no=shank)
            # ---- Save fig and insert ----
            fn_prefix = f'{water_res_num}_{sess_date}_{key["insertion_number"]}_{shank}_'
            fig_dict = save_figs((fig,), ('coronal_slice',), probe_dir, fn_prefix)
            plt.close('all')
            self.insert1({**key, **fig_dict, 'shank': shank})

# ============================= UNIT LEVEL ====================================


@schema
class UnitLevelEphysReport(dj.Computed):
    definition = """
    -> ephys.Unit
    ---
    unit_psth: filepath@report_store
    """

    # only units with UnitPSTH computed (delay-response task only),
    # and with InsertionLocation present
    key_source = (ephys.Unit & ephys.ProbeInsertion.InsertionLocation
                  & psth.UnitPsth & 'unit_quality != "all"')

    def make(self, key):
        if not ephys.check_unit_criteria(key):
            raise FailedUnitCriteriaError(f'Unit {key} did not meet selection criteria')

        water_res_num, sess_date = get_wr_sessdate(key)
        units_dir = store_stage / water_res_num / sess_date / str(key['insertion_number']) / 'units'
        units_dir.mkdir(parents=True, exist_ok=True)

        fig1 = unit_psth.plot_unit_psth(key)

        # ---- Save fig and insert ----
        fn_prefix = f'{water_res_num}_{sess_date}_{key["insertion_number"]}_{key["clustering_method"]}_u{key["unit"]:03}_'
        fig_dict = save_figs((fig1,), ('unit_psth',), units_dir, fn_prefix)

        plt.close('all')
        self.insert1({**key, **fig_dict})


@schema
class UnitLevelTrackingReport(dj.Computed):
    definition = """
    -> ephys.Unit
    ---
    unit_behavior: filepath@report_store
    """

    # only units from delay-response task with ingested tracking
    key_source = (ephys.Unit & tracking.Tracking & 'unit_quality != "all"'
                  & (experiment.Session
                     & (experiment.BehaviorTrial & 'task = "audio delay"')))

    def make(self, key):
        if not ephys.check_unit_criteria(key):
            raise FailedUnitCriteriaError(f'Unit {key} did not meet selection criteria')

        water_res_num, sess_date = get_wr_sessdate(key)
        units_dir = store_stage / water_res_num / sess_date / str(key['insertion_number']) / 'units'
        units_dir.mkdir(parents=True, exist_ok=True)

        fig1 = plt.figure(figsize=(16, 16))
        gs = GridSpec(4, 2)

        # 15 trials roughly in the middle of the session
        session = experiment.Session & key
        behavior_plot.plot_tracking(session, key, tracking_feature='jaw_y', xlim=(-0.5, 1),
                                    trial_offset=0.5, trial_limit=15,
                                    axs=np.array([fig1.add_subplot(gs[:3, col])
                                                  for col in range(2)]))

        axs = np.array([fig1.add_subplot(gs[-1, col], polar=True) for col in range(2)])
        behavior_plot.plot_unit_jaw_phase_dist(experiment.Session & key, key, axs=axs)
        [a.set_title('') for a in axs]

        fig1.subplots_adjust(wspace=0.4)
        fig1.subplots_adjust(hspace=0.6)

        # ---- Save fig and insert ----
        fn_prefix = f'{water_res_num}_{sess_date}_{key["insertion_number"]}_{key["clustering_method"]}_u{key["unit"]:03}_'
        fig_dict = save_figs((fig1,), ('unit_behavior',), units_dir, fn_prefix)

        plt.close('all')
        self.insert1({**key, **fig_dict})


@schema
class UnitMTLTrackingReport(dj.Computed):
    definition = """
    -> ephys.Unit
    ---
    unit_mtl_tracking: filepath@report_store
    """

    # only units with ingested tracking
    key_source = (ephys.Unit & tracking.Tracking
                  & oralfacial_analysis.JawTuning
                  & oralfacial_analysis.BreathingTuning
                  & oralfacial_analysis.WhiskerTuning)

    def make(self, key):
        if not ephys.check_unit_criteria(key):
            raise FailedUnitCriteriaError(f'Unit {key} did not meet selection criteria')

        water_res_num, sess_date = get_wr_sessdate(key)
        units_dir = store_stage / water_res_num / sess_date / str(key['insertion_number']) / 'units'
        units_dir.mkdir(parents=True, exist_ok=True)

        # this is a Multi-Target-Licking type of session (coming from RRig-MTL)
        # use plotting function from analysis_mtl
        fig1 = plt.figure(figsize=(16, 16))
        gs = GridSpec(3, 4)

        session = experiment.Session & key
        mtl_plot.plot_all_traces(
            session, key,
            tracking_feature='jaw_y',
            xlim=(0, 5),
            trial_offset=0, trial_limit=5,
            axs=np.array([fig1.add_subplot(gs[row_idx, col_idx])
                          for row_idx, col_idx in itertools.product(
                    range(3), range(3))]))

        mtl_plot.plot_jaw_tuning(key, axs=fig1.add_subplot(gs[0, 3], polar=True))
        mtl_plot.plot_breathing_tuning(key, axs=fig1.add_subplot(gs[1, 3], polar=True))

        fig1.subplots_adjust(wspace=0.2)
        fig1.subplots_adjust(hspace=0.6)

        # ---- Save fig and insert ----
        fn_prefix = f'{water_res_num}_{sess_date}_{key["insertion_number"]}_{key["clustering_method"]}_u{key["unit"]:03}_'
        fig_dict = save_figs((fig1,), ('unit_mtl_tracking',), units_dir, fn_prefix)

        plt.close('all')
        self.insert1({**key, **fig_dict})

# ============================= PROJECT LEVEL ====================================


@schema
class ProjectLevelProbeTrack(dj.Computed):
    definition = """
    -> experiment.Project
    ---
    tracks_plot: filepath@report_store
    track_count: int
    """

    key_source = experiment.Project & 'project_name = "MAP"'

    def make(self, key):
        proj_dir = store_stage

        sessions_probe_tracks, sessions_track_count = SessionLevelProbeTrack.fetch('probe_tracks', 'probe_track_count')
        track_count = sessions_track_count.sum().astype(int)

        probe_tracks_list = []
        for probe_tracks in sessions_probe_tracks:
            for shank_points in probe_tracks.values():
                probe_tracks_list.extend([shank_points] if isinstance(shank_points, np.ndarray) else shank_points)

        # ---- plot ----
        um_per_px = 20
        # fetch mesh
        vertices, faces = (ccf.AnnotatedBrainSurface
                           & 'annotated_brain_name = "Annotation_new_10_ds222_16bit_isosurf"').fetch1(
            'vertices', 'faces')
        vertices = vertices * um_per_px

        fig1 = plt.figure(figsize=(16, 12))
        fig1.suptitle('{} probe-tracks / {} probe-insertions'.format(track_count, len(ephys.ProbeInsertion())),
                      fontsize=24, y=0.05)

        for axloc, elev, azim in zip((221, 222, 223, 224), (65, 0, 90, 0), (-15, 0, 0, 90)):
            ax = fig1.add_subplot(axloc, projection='3d')
            ax.view_init(elev, azim)

            ax.grid(False)
            ax.invert_zaxis()

            ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2],
                            alpha=0.25, lw=0)

            for v in probe_tracks_list:
                ax.plot(v[:, 0], v[:, 2], v[:, 1], 'r')

            ax.set_title('Probe Track in CCF (um)', fontsize=16)

        # ---- Save fig and insert ----
        fn_prefix = (experiment.Project & key).fetch1('project_name') + '_'
        fig_dict = save_figs((fig1,), ('tracks_plot',), proj_dir, fn_prefix)

        plt.close('all')
        self.insert1({**key, **fig_dict, 'track_count': track_count})


# ---------- HELPER FUNCTIONS --------------

report_tables = [SessionLevelReport,
                 SessionLevelForagingSummary,
                 SessionLevelForagingLickingPSTH,
                 ProbeLevelReport,
                 ProbeLevelPhotostimEffectReport,
                 UnitLevelEphysReport,
                 UnitMTLTrackingReport,
                 UnitLevelTrackingReport,
                 SessionLevelCDReport,
                 SessionLevelProbeTrack,
                 ProjectLevelProbeTrack,
                 ProbeLevelDriftMap,
                 ProbeLevelCoronalSlice]


def get_wr_sessdate(key):
    water_res_num, session_datetime = (lab.WaterRestriction * experiment.Session.proj(
        session_datetime="cast(concat(session_date, ' ', session_time) as datetime)") & key).fetch1(
        'water_restriction_number', 'session_datetime')
    return water_res_num, datetime.strftime(session_datetime, '%Y%m%d_%H%M%S')


def save_figs(figs, fig_names, dir2save, prefix):
    fig_dict = {}
    for fig, figname in zip(figs, fig_names):
        fig_fp = dir2save / (prefix + figname + '.png')
        fig.tight_layout()
        fig.savefig(fig_fp)
        print(f'Generated {fig_fp}')
        fig_dict[figname] = fig_fp.as_posix()

    return fig_dict


def delete_outdated_session_plots():

    # ------------- SessionLevelProbeTrack ----------------

    sess_probe_hist = experiment.Session.aggr(histology.LabeledProbeTrack, probe_hist_count='count(*)')
    oudated_sess_probe = SessionLevelProbeTrack * sess_probe_hist & 'probe_track_count != probe_hist_count'

    uuid_bytes = (SessionLevelProbeTrack & oudated_sess_probe.proj()).proj(ub='(session_tracks_plot)').fetch('ub')

    if len(uuid_bytes):
        ext_keys = [{'hash': uuid.UUID(bytes=uuid_byte)} for uuid_byte in uuid_bytes]

        with dj.config(safemode=False):
            with SessionLevelProbeTrack.connection.transaction:
                # delete the outdated Probe Tracks
                (SessionLevelProbeTrack & oudated_sess_probe.proj()).delete()
                # delete from external store
                (schema.external['report_store'] & ext_keys).delete(delete_external_files=True)
                print('{} outdated SessionLevelProbeTrack deleted'.format(len(uuid_bytes)))
    else:
        print('All SessionLevelProbeTrack are up-to-date')

    # ------------- SessionLevelCDReport ----------------

    sess_probe = experiment.Session.aggr(ephys.ProbeInsertion, current_probe_count='count(*)')
    oudated_sess_probe = SessionLevelCDReport * sess_probe & 'cd_probe_count != current_probe_count'

    uuid_bytes = (SessionLevelCDReport & oudated_sess_probe.proj()).proj(ub='(coding_direction)').fetch('ub')

    if len(uuid_bytes):
        ext_keys = [{'hash': uuid.UUID(bytes=uuid_byte)} for uuid_byte in uuid_bytes]

        with dj.config(safemode=False):
            with SessionLevelCDReport.connection.transaction:
                # delete the outdated Probe Tracks
                (SessionLevelCDReport & oudated_sess_probe.proj()).delete()
                # delete from external store
                (schema.external['report_store'] & ext_keys).delete(delete_external_files=True)
                print('{} outdated SessionLevelCDReport deleted'.format(len(uuid_bytes)))
    else:
        print('All SessionLevelCDReport are up-to-date')


def delete_outdated_project_plots(project_name='MAP'):
    if {'project_name': project_name} not in ProjectLevelProbeTrack.proj():
        return

    plotted_track_count = (ProjectLevelProbeTrack & {'project_name': project_name}).fetch1('track_count')
    latest_track_count = SessionLevelProbeTrack.fetch('probe_track_count').sum().astype(int)

    if plotted_track_count != latest_track_count:
        uuid_byte = (ProjectLevelProbeTrack & {'project_name': project_name}).proj(
            ub='CONCAT(tracks_plot , "")').fetch1('ub')
        print('project_name', project_name, 'uuid_byte:', str(uuid_byte))
        ext_key = {'hash': uuid.UUID(bytes=uuid_byte)}

        with dj.config(safemode=False):
            with ProjectLevelProbeTrack.connection.transaction:
                # delete the outdated Probe Tracks
                (ProjectLevelProbeTrack & {'project_name': project_name}).delete()
                # delete from external store
                (schema.external['report_store'] & ext_key).delete(delete_external_files=True)
                print('Outdated ProjectLevelProbeTrack deleted')
    else:
        print('ProjectLevelProbeTrack is up-to-date')
