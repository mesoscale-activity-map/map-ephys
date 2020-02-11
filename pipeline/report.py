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

import io
from PIL import Image
import itertools

from pipeline import experiment, ephys, psth, tracking, lab, histology, ccf
from pipeline.plot import behavior_plot, unit_characteristic_plot, unit_psth, histology_plot
from pipeline import get_schema_name
from pipeline.plot.util import _plot_with_sem, _jointplot_w_hue
from pipeline.util import _get_trial_event_times

import warnings
warnings.filterwarnings('ignore')


schema = dj.schema(get_schema_name('report'))

os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = "TRUE"

store_stage = pathlib.Path(dj.config['stores']['report_store']['stage'])

if dj.config['stores']['report_store']['protocol'] == 's3':
    store_location = (pathlib.Path(dj.config['stores']['report_store']['bucket'])
                      / pathlib.Path(dj.config['stores']['report_store']['location']))
    store_location = 'S3: ' + str(store_location)
else:
    store_location = pathlib.Path(dj.config['stores']['report_store']['location'])

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
            behavior_plot.plot_photostim_effect(key, stim_key, axs=ax, title=stim_loc)
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
        self.insert1({**key, **fig_dict})


@schema
class SessionLevelProbeTrack(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    session_tracks_plot: filepath@report_store
    probe_tracks: longblob
    """

    @property
    def key_source(self):
        # Only process Session with ProbeTrack histology available
        sess_probes = experiment.Session.aggr(ephys.ProbeInsertion, probe_count='count(*)')
        sess_probe_hist = experiment.Session.aggr(histology.LabeledProbeTrack, probe_hist_count='count(*)')
        return sess_probes * sess_probe_hist & 'probe_count = probe_hist_count'

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
        self.insert1({**key, **fig_dict, 'probe_tracks': probe_tracks})


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
        unit_characteristic_plot.plot_unit_bilateral_photostim_effect(
            probe_insertion, clustering_method=key['clustering_method'], axs=axs[-1])

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
        fig1, axs = plt.subplots(1, 1 + len(stim_locs), figsize=(16, 6))
        for pos, stim_loc in enumerate(stim_locs):
            unit_characteristic_plot.plot_psth_photostim_effect(units,
                                                                condition_name_kw=[stim_loc.lower()],
                                                                axs=np.array([axs[0], axs[pos + 1]]))
            if pos < len(stim_locs) - 1:
                axs[0].clear()

        [a.set_title(title.replace('_', ' ').upper()) for a, title in zip(axs, ['control'] + list(stim_locs))]

        # ---- Save fig and insert ----
        fn_prefix = f'{water_res_num}_{sess_date}_{key["insertion_number"]}_{key["clustering_method"]}_'
        fig_dict = save_figs((fig1,), ('group_photostim',), probe_dir, fn_prefix)

        plt.close('all')
        self.insert1({**key, **fig_dict})

# ============================= UNIT LEVEL ====================================


@schema
class UnitLevelEphysReport(dj.Computed):
    definition = """
    -> ephys.Unit
    ---
    unit_psth: filepath@report_store
    """

    # only units UnitPSTH computed, and with InsertionLocation present
    key_source = ephys.Unit & ephys.ProbeInsertion.InsertionLocation & psth.UnitPsth & 'unit_quality != "all"'

    def make(self, key):
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

    # only units with ingested tracking
    key_source = ephys.Unit & tracking.Tracking & 'unit_quality != "all"'

    def make(self, key):
        water_res_num, sess_date = get_wr_sessdate(key)
        units_dir = store_stage / water_res_num / sess_date / str(key['insertion_number']) / 'units'
        units_dir.mkdir(parents=True, exist_ok=True)

        fig1 = plt.figure(figsize = (16, 16))
        gs = GridSpec(4, 2)

        # 15 trials roughly in the middle of the session
        session = experiment.Session & key
        behavior_plot.plot_tracking(session, key, tracking_feature='jaw_y', xlim=(-0.5, 1),
                                    trial_offset=0.5, trial_limit=15, axs=np.array([fig1.add_subplot(gs[:3, col])
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

# ============================= PROJECT LEVEL ====================================


@schema
class ProjectLevelProbeTrack(dj.Computed):
    definition = """
    -> experiment.Project
    ---
    tracks_plot: filepath@report_store
    session_count: int
    """

    key_source = experiment.Project & 'project_name = "MAP"'

    def make(self, key):
        proj_dir = store_stage

        session_count = len(SessionLevelProbeTrack())

        sessions_probe_tracks = SessionLevelProbeTrack.fetch('probe_tracks')

        probe_tracks_list = list(itertools.chain(*[[v for v in probe_tracks.values()]
                                                   for probe_tracks in sessions_probe_tracks]))

        # ---- plot ----
        um_per_px = 20
        # fetch mesh
        vertices, faces = (ccf.AnnotatedBrainSurface
                           & 'annotated_brain_name = "Annotation_new_10_ds222_16bit_isosurf"').fetch1(
            'vertices', 'faces')
        vertices = vertices * um_per_px

        fig1 = plt.figure(figsize=(16, 12))

        for axloc, elev, azim in zip((221, 222, 223, 224), (65, 0, 90, 0), (-15, 0, 0, 90)):
            ax = fig1.add_subplot(axloc, projection='3d')
            ax.view_init(elev, azim)

            ax.grid(False)
            ax.invert_zaxis()

            ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2],
                            alpha = 0.25, lw = 0)

            for v in probe_tracks_list:
                ax.plot(v[:, 0], v[:, 2], v[:, 1], 'r')

            ax.set_title('Probe Track in CCF (um)')

        # ---- Save fig and insert ----
        fn_prefix = (experiment.Project & key).fetch1('project_name') + '_'
        fig_dict = save_figs((fig1,), ('tracks_plot',), proj_dir, fn_prefix)

        plt.close('all')
        self.insert1({**key, **fig_dict, 'session_count': session_count})


# ---------- HELPER FUNCTIONS --------------

report_tables = [SessionLevelReport,
                 ProbeLevelReport,
                 ProbeLevelPhotostimEffectReport,
                 UnitLevelEphysReport,
                 UnitLevelTrackingReport,
                 SessionLevelCDReport,
                 SessionLevelProbeTrack,
                 ProjectLevelProbeTrack]


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


def delete_outdated_probe_tracks(project_name='MAP'):
    if {'project_name': project_name} not in ProjectLevelProbeTrack.proj():
        return

    sess_count = (ProjectLevelProbeTrack & {'project_name': project_name}).fetch1('session_count')
    latest_sess_count = len(SessionLevelProbeTrack())

    if sess_count != latest_sess_count:
        uuid_byte = (ProjectLevelProbeTrack & {'project_name': project_name}).proj(ub='(tracks_plot)').fetch1('ub')
        ext_key = {'hash': uuid.UUID(bytes=uuid_byte)}

        with dj.config(safemode=False) as cfg:
            with ProjectLevelProbeTrack.connection.transaction:
                # delete the outdated Probe Tracks
                (ProjectLevelProbeTrack & {'project_name': project_name}).delete()
                # delete from external store
                (schema.external['report_store'] & ext_key).delete(delete_external_files=True)
                print('Outdated ProjectLevelProbeTrack deleted')

    else:
        print('ProjectLevelProbeTrack is up-to-date')
