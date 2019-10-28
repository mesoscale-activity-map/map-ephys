#! /usr/bin/env python
# map-ephys interative shell

import time
import os
import pipeline.shell as shell


settings = {'reserve_jobs': True, 'suppress_errors': True, 'display_progress': True, 'order': 'random'}


def main():
    shell.logsetup(os.environ.get('MAP_LOGLEVEL', 'INFO'))

    print('=========================================================================')
    print('========================== AUTO POPULATE =======================')
    print('=========================================================================')
    print(f'\n{settings}\n')

    while True:
        shell.actions['populate-psth']()
        shell.actions['generate-report']()

        time.sleep(1)


if __name__ == '__main__':
    main()
