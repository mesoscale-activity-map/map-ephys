
#123456789#123456789#123456789#123456789#123456789#123456789#123456789#123456789

import os
from datajoint import config as config

#
# Utilities
#

def run_system_cmd(cmdstr):
    if os.system(cmdstr):
        raise Exception('command "{}" failed'.format(cmdstr))


def setup():
    # safety hack to prevent dropping live databasess
    if 'do_unittest' not in config or config['do_unittest'] is not True:
        raise Exception("tests skipped - dj.config not testing configuration") 


#
# Actual Tests
#


def test_run_system_cmd():
    run_system_cmd('echo')

    try:
        run_system_cmd('sdajk jfakjdkljfk djaklj adkljf ka')
    except Exception:
        return True

    raise Exception("bad command didn't yield exception")


def test_mock():
    # TODO: should be run with safeguards, or perhaps mock should have safeguards
    run_system_cmd('./mock.py')


def test_behavior_ingest():
    # TODO: should be run with safeguards, or perhaps mock should have safeguards
    run_system_cmd('./mapshell.py populateB')


def test_ephys_ingest():
    # TODO: should be run with safeguards, or perhaps mock should have safeguards
    run_system_cmd('./mapshell.py populateE')
