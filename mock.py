#! /usr/bin/env python

import sys
from importlib import reload

import datajoint as dj

import lab
import ccf
import ephys
import experiment


def dropdbs():
    print('dropping databases')
    for a in range(3):
        for d in ['ingest', 'ccf', 'ephys', 'experiment', 'lab', 'prototype']:
            try:
                schema = dj.schema(dj.config['%s.database' % d])
                schema.drop(force=True)
            except Exception as e:
                print('error dropping {d} in attempt {a}: {e}'
                      .format(d=d, a=a, e=str(e)))
                pass


def mockdata():
    print('populating with mock data')
    reload(ccf)
    reload(lab)
    reload(experiment)
    reload(ephys)
    try:
        # TODO: these should be loaded in a more 'official' way
        lab.Person().insert1({
            'username': 'daveliu',
            'fullname': 'Dave Liu'},
            skip_duplicates = True
        )
        lab.ModifiedGene().insert1({
            'gene_modification': 'VGAT-Chr2-EYFP Jax',
            'description': 'VGAT'},
            skip_duplicates = True
        )
        lab.Subject().insert1({
            'subject_id': 399752,
            'username': 'daveliu',
            'cage_number': 145375,
            'date_of_birth': '2017-08-03',
            'sex': 'M',
            'animal_source': 'Jackson labs'},
            skip_duplicates = True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 399752,
            'water_restriction_number': 'dl7',
            'cage_number': 148861,
            'wr_start_date': '2017-11-07',
            'wr_start_weight': 20.5},
            skip_duplicates = True
        )
        lab.Subject().insert1({
            'subject_id': 397853,
            'username': 'daveliu',
            'cage_number': 144545,
            'date_of_birth': '2017-08-03',
            'sex':  'M',
            'animal_source': 'Jackson labs'},
            skip_duplicates = True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 397853,
            'water_restriction_number': 'dl14',
            'cage_number': 149595,
            'wr_start_date': '2017-11-27',
            'wr_start_weight': 20.5},
            skip_duplicates = True
        )
        lab.Subject().insert1({
            'subject_id': 123457,
            'username': 'daveliu',
            'cage_number': 145375,
            'date_of_birth': '2017-08-03',
            'sex': 'M',
            'animal_source': 'Jackson labs'},
            skip_duplicates = True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 123457,
            'water_restriction_number': 'tw5',
            'cage_number': 148861,
            'wr_start_date': '2017-11-07',
            'wr_start_weight': 20.5},
            skip_duplicates = True
        )
        lab.Rig().insert1({
            'rig': 'TRig1',
            'room': '2w.334',
            'rig_description': 'Training rig 1'},
            skip_duplicates = True
        )
        lab.Rig().insert1({
            'rig': 'TRig2',
            'room': '2w.334',
            'rig_description': 'Training rig 2'},
            skip_duplicates = True
        )
        lab.Rig().insert1({
            'rig': 'TRig3',
            'room': '2w.334',
            'rig_description': 'Training rig 3'},
            skip_duplicates = True
        )
        lab.Rig().insert1({
            'rig': 'RRig',
            'room': '2w.334',
            'rig_description': 'Recording rig'},
            skip_duplicates = True
        )
        lab.Rig().insert1({
            'rig': 'Ephys1',
            'room': '2w.334',
            'rig_description': 'Recording computer'},
            skip_duplicates = True
        )
    except Exception as e:
        print("error creating mock data: {e}".format(e=e), file=sys.stderr)
        raise

if __name__ == '__main__':
    dropdbs()
    mockdata()
