

import logging
import os.path
import unittest

from datetime import datetime

import datajoint as dj

from pipeline.globus import GlobusStorageManager


log = logging.getLogger(__name__)

# todo: expand urls to also build remote endpoint url for cp test
# local agent endpoint
test_ep = dj.config['globus.local_endpoint']

# subdirectory root for tests within local endpoint
test_ep_subdir = dj.config['globus.local_endpoint_subdir']

# globus 'url' and local path for test subdirectory
test_ep_url = '{}:{}'.format(test_ep, test_ep_subdir)
test_ep_local_path = dj.config['globus.local_endpoint_local_path']

# directory used for this test run
test_dir = datetime.now().strftime('test-%Y-%m-%d_%H:%M:%S')

# local path to directory used for this test run
mkdir_local_path = os.path.join(test_ep_local_path, test_dir)

# globus url within local endpoint to this run's test directory
mkdir_url = '{}/{}'.format(test_ep_url, test_dir)



def test_globus_constructor():
    ''' test globus constructor '''
    gsm = GlobusStorageManager()
    return gsm


def test_globus_login():
    ''' test globus login '''
    # TODO: not implemented due to raw_input;
    # ensure test config has valid 'globus_token'; to do this, use following:
    #
    #   >>> gsm = GlobusStorageManager()
    #   >>> gsm.login()
    #
    pass


def test_globus_activate_ep():
    ''' test adding an endpoint '''
    global test_ep
    gsm = GlobusStorageManager()

    try:
        gsm.activate_endpoint(None)
    except Exception:
        pass

    gsm.activate_endpoint(test_ep)


def test_globus_ls():
    ''' test globus ls '''
    global test_ep, test_ep_local_path, test_ep_url

    test_fname = 'test.txt'
    test_fpath = os.path.join(test_ep_local_path, test_fname)

    if not os.path.exists(test_fpath):
        with open(test_fpath, 'w') as f:
            f.write('test file for ls')

    gsm = GlobusStorageManager()
    gsm.activate_endpoint(test_ep)

    file_list = gsm.ls(test_ep_url)
    log.debug('test_globus_ls(): file_list: {}'.format(file_list))

    os.remove(test_fpath)

    files = [e['name'] for e in file_list['DATA']]
    if test_fname not in files:
        raise Exception("{} not in entries".format(test_fname))


def test_globus_mkdir():
    ''' test globus mkdir '''
    global test_ep, test_dir, mkdir_url
    gsm = GlobusStorageManager()
    gsm.activate_endpoint(test_ep)

    log.debug('test_globus_mkdir: creating {}'.format(test_dir))

    mkcode = gsm.mkdir(mkdir_url)
    log.debug('mkdir {} returned {}'.format(mkdir_url, mkcode))


@unittest.skip('need reconfig: subscription reqd for personal<->personal xfer')
def test_globus_cp():
    ''' test globus cp '''
    global test_dir, mkdir_local_path, mkdir_url

    srcfp = os.path.join(mkdir_local_path, 'testfile.src')

    with open(srcfp, 'w') as f:
        f.write('test file for {}'.format(test_dir))

    gsm = GlobusStorageManager()
    gsm.activate_endpoint(test_ep)

    srcp = '{}/{}'.format(mkdir_url, 'testfile.src')
    dstp = '{}/{}'.format(mkdir_url, 'testfile.dst')

    if not gsm.cp(srcp, dstp):
        raise Exception('cp from {} to {} failed.'.format(srcp, dstp))


@unittest.skip('need reconfig: subscription reqd for personal<->personal xfer')
def test_globus_rename():
    ''' test globus rename '''
    global mkdir_path, test_ep
    srcp = '{}/{}'.format(mkdir_url, 'testfile.dst')
    dstp = '{}/{}'.format(mkdir_url, 'testfile.end')

    gsm = GlobusStorageManager()
    gsm.activate_endpoint(test_ep)

    if not gsm.rename(srcp, dstp):
        raise Exception('rename from {} to {} failed.'.format(srcp, dstp))


def test_globus_rmdir():
    ''' test globus rmdir '''
    global test_ep, mkdir_url
    gsm = GlobusStorageManager()
    log.debug('test_globus_rmdir: removing {}'.format(mkdir_url))

    gsm.activate_endpoint(test_ep)

    subscription_fixed_and_cp_tested = False
    if subscription_fixed_and_cp_tested:
        if gsm.rmdir(mkdir_url):
            raise Exception('rmdir {} should have failed.'.format(mkdir_url))

    if not gsm.rmdir(mkdir_url, recursive=True):
        raise Exception('rmdir of {} failed.'.format(mkdir_url))
