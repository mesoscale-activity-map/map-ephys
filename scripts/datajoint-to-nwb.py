import os

import json
import uuid

from uuid import UUID
from datetime import date, datetime

import numpy as np
from pipeline import lab, publication, experiment, ephys, psth, ccf


for session_key in acquisition.Session.fetch('KEY'):
    this_session = (acquisition.Session & session_key).fetch1()
    # =============== General ====================
    # -- NWB file - a NWB2.0 file for each session
    nwbfile = NWBFile(
        session_description=this_session['session_note'],
        identifier='_'.join(
            [this_session['subject_id'],
             this_session['session_time'].strftime('%Y-%m-%d_%H-%M-%S')]),
        session_start_time=this_session['session_time'],
        file_create_date=datetime.now(tzlocal()),
        experimenter='; '.join((acquisition.Session.Experimenter
                                & session_key).fetch('experimenter')),
        institution='Janelia Research Campus',
        related_publications='doi:10.1038/nature22324')
    # -- subject
    subj = (subject.Subject & session_key).fetch1()
    nwbfile.subject = pynwb.file.Subject(
        subject_id=this_session['subject_id'],
        description=subj['subject_description'],
        genotype=' x '.join((subject.Subject.Allele
                             & session_key).fetch('allele')),
        sex=subj['sex'],
        species=subj['species'])
