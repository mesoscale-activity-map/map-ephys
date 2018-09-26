#! /usr/bin/env python
# map-ephys interative shell

import os
import sys
import logging
import warnings
from code import interact

import datajoint as dj

from pipeline import ephys
from pipeline import lab
from pipeline import experiment
from pipeline import ccf
from pipeline import publication

log = logging.getLogger(__name__)
__all__ = [ephys, lab, experiment, ccf, publication]

[ dj ]  # NOQA flake8 

warnings.simplefilter(action='ignore', category=FutureWarning)


def usage_exit():
    print("usage: {p} [{c}]"
          .format(p=os.path.basename(sys.argv[0]),
                  c='|'.join(list(actions.keys()))))
    sys.exit(0)


def logsetup(*args):
    logging.basicConfig(level=logging.ERROR)
    log.setLevel(logging.DEBUG)
    logging.getLogger('pipeline.ingest.behavior').setLevel(logging.DEBUG)
    logging.getLogger('pipeline.ingest.ephys').setLevel(logging.DEBUG)
    logging.getLogger('pipeline.publication').setLevel(logging.DEBUG)


def populateB(*args):
    from pipeline.ingest import behavior as ingest_behavior
    ingest_behavior.BehaviorIngest().populate(display_progress=True)


def populateE(*args):
    from pipeline.ingest import ephys as ingest_ephys
    ingest_ephys.EphysIngest().populate(display_progress=True)


def publish(*args):
    publication.ArchivedRawEphysTrial.populate()


def shell(*args):
    interact('map shell.\n\nschema modules:\n\n  - {m}\n'
             .format(m='\n  - '.join(
                 '.'.join(m.__name__.split('.')[1:]) for m in __all__)),
             local=globals())


def erd(*args):
    for mod in (ephys, lab, experiment, ccf,):
        modname = str().join(mod.__name__.split('.')[1:])
        fname = os.path.join('pipeline', '{}.png'.format(modname))
        print('saving', fname)
        dj.ERD(mod, context={modname: mod}).save(fname)


actions = {
    'populateB': populateB,
    'populateE': populateE,
    'publish': publish,
    'shell': shell,
    'erd': erd,
}

if __name__ == '__main__':

    if len(sys.argv) < 2 or sys.argv[1] not in actions:
        usage_exit()

    logsetup()

    action = sys.argv[1]
    actions[action](sys.argv[2:])
