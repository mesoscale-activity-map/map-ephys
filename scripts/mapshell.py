#! /usr/bin/env python
# map-ephys interative shell

import os
import sys
import logging

import datajoint as dj

import pipeline.shell as shell

log = logging.getLogger(__name__)


if __name__ == '__main__':

    if len(sys.argv) < 2 or sys.argv[1] not in shell.actions:
        shell.usage_exit()

    shell.logsetup(os.environ.get(
        'MAP_LOGLEVEL', dj.config.get('loglevel', 'INFO')))

    try:
        action = sys.argv[1]
        shell.actions[action][0](*sys.argv[2:])
    except Exception:
        log.exception("action '{}' encountered an exception:".format(action))
