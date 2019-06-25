import numpy as np
import datajoint as dj
import itertools
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from pipeline import experiment, ephys, ccf, histology


def plot_probe_tracks(session_key, ax=None):
    mm_per_px = 0.02
    # fetch mesh
    vertices, faces = (ccf.AnnotatedBrainSurface
                       & 'annotated_brain_name = "Annotation_new_10_ds222_16bit_isosurf"').fetch1(
        'vertices', 'faces')
    vertices = vertices * mm_per_px

    probe_tracks = {}
    for probe_insert in (ephys.ProbeInsertion & session_key).fetch('KEY'):
        points = (histology.LabeledProbeTrack.Point & probe_insert).fetch('ccf_x', 'ccf_y', 'ccf_z', order_by='"order"')
        probe_tracks[probe_insert['insertion_number']] = np.vstack(zip(*points))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    assert isinstance(ax, Axes3D)

    # cosmetic
    plt.gca().patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((0, 0, 0, 1.0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 1.0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 1.0))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.invert_zaxis()

    for k, v in probe_tracks.items():
        v = v * mm_per_px
        ax.plot(v[:, 0], v[:, 2], v[:, 1], 'r', label = f'probe {k}')

    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2],
                    alpha=0.2, lw=0)


