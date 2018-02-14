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
        lab.Animal().insert1({
            'animal': 399752,
            'dob':  '2017-08-01'
        })
        lab.AnimalWaterRestriction().insert1({
            'animal': 399752,
            'water_restriction': 'dl7'
        })
        lab.Animal().insert1({
            'animal': 397853,
            'dob':  '2017-08-01'
        })
        lab.AnimalWaterRestriction().insert1({
            'animal': 397853,  # bogus id
            'water_restriction': 'dl14'
        })
        lab.Animal().insert1({
            'animal': 123457,  # bogus id
            'dob':  '2017-08-01'
        })
        lab.AnimalWaterRestriction().insert1({
            'animal': 123457,  # bogus id
            'water_restriction': 'tw5'
        })

        lab.Person().insert1({
            'username': 'daveliu',
            'fullname': 'Dave Liu'
        })
        lab.Rig().insert1({
            'rig': 'TRig1',
            'rig_description': 'Training rig 1',
        })
        lab.Rig().insert1({
            'rig': 'TRig2',
            'rig_description': 'Training rig 2',
        })
        lab.Rig().insert1({
            'rig': 'TRig3',
            'rig_description': 'Training rig 3',
        })
        lab.Rig().insert1({
            'rig': 'RRig',
            'rig_description': 'Recording rig',
        })
        lab.Rig().insert1({
            'rig': 'EPhys1',
            'rig_description': 'Ephys rig 1',
        })
    except Exception as e:
        print("error creating mock data: {e}".format(e=e), file=sys.stderr)
        raise


if __name__ == '__main__':
    dropdbs()
    mockdata()
