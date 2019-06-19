#! /usr/bin/env python
# map-ephys interative shell

import os
import sys
import pipeline.shell as shell


if __name__ == '__main__':

    if len(sys.argv) < 2 or sys.argv[1] not in shell.actions:
        shell.usage_exit()

    shell.logsetup(os.environ.get('MAP_LOGLEVEL', 'INFO'))

    action = sys.argv[1]
    shell.actions[action](sys.argv[2:])
