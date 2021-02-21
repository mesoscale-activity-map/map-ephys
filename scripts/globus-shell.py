#! /usr/bin/env python

import sys
import logging
import posixpath as path

import datajoint as dj

from pipeline.globus import GlobusStorageManager
from pipeline.publication import GlobusStorageLocation

log = logging.getLogger(__name__)


class GlobusShell:
    def __init__(self, env={}):
        self._env = env                         # Environment (grab bag atm)
        self._gsm = GlobusStorageManager()      # Globus Storage Manager
        self._aep = set()                       # Active Endpoints
        self._cep = None                        # Current Endpoint
        self._cwd = '/'                         # Current Directory

    def _getpath(self, ep=None, wd=None):

        ep = ep if ep else self._cep
        wd = wd if wd else self._cwd

        if ep == self._cep:
            wd = wd if wd.startswith('/') else path.join(self._cwd, wd)
        else:
            wd = path.join('/', wd)

        return ep, wd

    def env(self):
        print(self._env)

    def cd(self, ep=None, wd='/'):
        ep, wd = self._getpath(ep, wd)

        log.debug('cd: {}:{}'.format(ep, wd))

        if ep not in self._aep:
            self._gsm.activate_endpoint(ep)
            self._aep = self._aep.union({(ep)})

        self._cep = ep
        self._cwd = wd

    def pwd(self):
        print('{}:{}'.format(self._cep, self._cwd))

    def mkdir(self, ep, wd):
        ep, wd = self._getpath(ep, wd)

        log.debug('mkdir: {}:{}'.format(ep, wd))

        return self._gsm.mkdir('{}:{}'.format(ep, wd))

    def ls(self, ep=None, wd=None):
        ep, wd = self._getpath(ep, wd)

        log.debug('ls: {}:{}'.format(ep, wd))

        lsdat = self._gsm.ls('{}:{}'.format(ep, wd))

        for i in iter(lsdat):
            if i['type'] == 'file':
                print('f: {}'.format(i['name']))
            else:
                print('d: {}'.format(i['name']))

        return True

    def rm(self, ep, wd):
        ep, wd = self._getpath(ep, wd)

        log.debug('rm: {}:{}'.format(ep, wd))

        return self._gsm.rm('{}:{}'.format(ep, wd))

    def rmdir(self, ep, wd):
        ep, wd = self._getpath(ep, wd)

        log.debug('rmdir: {}:{}'.format(ep, wd))

        return self._gsm.rmdir('{}:{}'.format(ep, wd))

    def rm_r(self, ep, wd):
        # XXX: drop when shell gets 'args' support & add -r flag
        ep, wd = self._getpath(ep, wd)

        log.debug('rm_r: {}:{}'.format(ep, wd))

        return self._gsm.rm('{}:{}'.format(ep, wd), recursive=True)

    def mv(self, ep1, wd1, ep2, wd2):

        ep1, wd1 = self._getpath(ep1, wd1)
        ep2, wd2 = self._getpath(ep2, wd2)

        if ep1 != ep2:
            return False

        log.debug('mv: {}:{} -> {}:{}'.format(ep1, wd1, ep2, wd2))

        return self._gsm.rename('{}:{}'.format(ep1, wd1),
                                '{}:{}'.format(ep2, wd2))

    def cp(self, ep1, wd1, ep2, wd2):
        # XXX: delete_destination_xtra for mirroring?

        ep1, wd1 = self._getpath(ep1, wd1)
        ep2, wd2 = self._getpath(ep2, wd2)

        log.debug('cp: {}:{} -> {}:{}'.format(ep1, wd1, ep2, wd2))

        return self._gsm.cp('{}:{}'.format(ep1, wd1),
                            '{}:{}'.format(ep2, wd2))

    def cp_r(self, ep1, wd1, ep2, wd2):

        ep1, wd1 = self._getpath(ep1, wd1)
        ep2, wd2 = self._getpath(ep2, wd2)

        log.debug('cp_r: {}:{} -> {}:{}'.format(ep1, wd1, ep2, wd2))

        return self._gsm.cp('{}:{}'.format(ep1, wd1), 
                            '{}:{}'.format(ep2, wd2),
                            recursive=True)

    def find(self, ep=None, wd=None):
        ep, wd = self._getpath(ep, wd)

        log.debug('find: {}:{}'.format(ep, wd))

        for ep, dirname, node in self._gsm.fts('{}:{}'.format(ep, wd)):

            if node['DATA_TYPE'] == 'file':
                t, basename = 'f', node['name']
            else:
                t, basename = 'd', None

            print('{}: {}:{}/{}'.format(t, ep, dirname, basename)
                  if t == 'f' else '{}: {}:{}'.format(t, ep, dirname))

    def sh(self, prompt='globus% '):
        cmds = set(('env', 'pwd', 'cd', 'ls', 'find', 'cp', 'cp_r', 'mkdir',
                    'mv', 'rm', 'rmdir', 'rm_r'))

        while True:
            try:
                data = input(prompt).split(' ')
            except EOFError:
                return

            cmd, args = data[0], data[1:]

            log.debug('cmd input: {} -> {}, {}'.format(data, cmd, args))

            if not cmd or cmd.startswith('#'):
                continue
            elif cmd in cmds:
                try:
                    # TODO: need 'flags' handling so e.g. 'cp -r' set recursive
                    args = (i.split(':') if ':' in i else (None, i)
                            for i in args)

                    ret = getattr(self, cmd)((*[j for i in args for j in i]))

                    if ret not in {True, None}:
                        print(ret)

                except Exception as e:
                    print(repr(e))
            else:
                print('commands: \n  - {}'.format('\n  - '.join(cmds)))


if __name__ == '__main__':

    log.setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.ERROR)

    leps = dj.config.get('custom', {}).get('globus.storage_locations', None)
    peps = GlobusStorageLocation().fetch(as_dict=True)

    sh = GlobusShell({'local_endpoints': leps, 'publication_endpoints': peps})

    if len(sys.argv) > 1:
        sys.stdin = open(sys.argv[1], 'r')
        exit(sh.sh(prompt=''))
    else:
        exit(sh.sh())
