#! /usr/bin/env python

import os
import sys
import logging
from code import interact

import datajoint as dj

import ephys
import lab
import experiment
import ccf
import ingest

log = logging.getLogger(__name__)
__all__ = [ephys, lab, experiment, ccf, ingest]
[ dj ]  # NOQA flake8 


def usage_exit():
    print("usage: {p} [discover|populate|shell]"
          .format(p=os.path.basename(sys.argv[0])))
    sys.exit(0)


def logsetup(*args):
    logging.basicConfig(level=logging.ERROR)
    log.setLevel(logging.INFO)
    logging.getLogger('ingest').setLevel(logging.INFO)


def discover(*args):
    ingest.SessionDiscovery().populate()


def populate(*args):
    ingest.RigDataFile().populate()  # populate overridden; jobs n/a & ni
    ingest.BehaviorIngest().populate(
        reserve_jobs=True, suppress_errors=True)
    ephys.ElectrodeGroup().insert1(
        (123456, 1, 1, '15131808323',
         './data_all/RRig/real5ap_imec3_opt3_jrc.mat'))
    # note: behavior / electrode group ingest a prerequsite for this to work
    ephys.ElectrodeGroup().make(ephys.ElectrodeGroup().fetch1(dj.key))
    ephys.Ephys().populate(
        reserve_jobs=True, suppress_errors=True)


def shell(*args):
    interact('map shell.\n\nschema modules:\n\n  - {m}\n'
             .format(m='\n  - '.join(str(m.__name__) for m in __all__)),
             local=globals())


actions = {
    'shell': shell,
    'discover': discover,
    'populate': populate,
}


if __name__ == '__main__':

    if len(sys.argv) < 2 or sys.argv[1] not in actions:
        usage_exit()

    logsetup()

    action = sys.argv[1]
    actions[action](sys.argv[2:])
