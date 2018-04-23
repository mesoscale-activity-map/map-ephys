#! /usr/bin/env python
#shell script to discover and populate the sessions

import os
import sys
import logging
import warnings
from code import interact

import datajoint as dj

import ephys
import lab
import experiment
import ccf
import ingestBehavior
import ingestEphys

log = logging.getLogger(__name__)
__all__ = [ephys, lab, experiment, ccf, ingestBehavior, ingestEphys]
[ dj ]  # NOQA flake8 

warnings.simplefilter(action='ignore', category=FutureWarning)

def usage_exit():
    print("usage: {p} [discover|populate|shell]"
          .format(p=os.path.basename(sys.argv[0])))
    sys.exit(0)

def logsetup(*args):
    logging.basicConfig(level=logging.ERROR)
    log.setLevel(logging.DEBUG)
    logging.getLogger('ingest').setLevel(logging.DEBUG)

def discover(*args):
    ingestBehavior.SessionDiscovery().populate()

def populateB(*args):
    ingestBehavior.BehaviorIngest().populate()

def populateE(*args):
    ingestEphys.EphysIngest().populate()

def shell(*args):
    interact('map shell.\n\nschema modules:\n\n  - {m}\n'
             .format(m='\n  - '.join(str(m.__name__) for m in __all__)),
             local=globals())

actions = {
    'shell': shell,
    'discover': discover,
    'populateB': populateB,
    'populateE': populateE
}

if __name__ == '__main__':

    if len(sys.argv) < 2 or sys.argv[1] not in actions:
        usage_exit()

    logsetup()

    action = sys.argv[1]
    actions[action](sys.argv[2:])