#! /usr/bin/env python

import sys
import logging

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

    def env(self):
        print(self._env)

    def pwd(self):
        print('{}:{}'.format(self._cep, self._cwd))

    def cd(self, ep=None, path=None):
        ep = ep if ep else self._cep
        path = (path if (ep == self._cep and path.startswith('/'))
                else '/'.join((self._cwd, path)))

        if ep not in self._aep:
            self._gsm.activate_endpoint(ep)
            self._aep = self._aep.union({(ep)})

        self._cep = ep
        self._cwd = '/' + path.lstrip('/')

    def ls(self, ep=None, path=None):

        ep = ep if ep else self._cep
        path = path if path else self._cwd

        lsdat = self._gsm.ls('{}:{}'.format(ep, path))
        for i in iter(lsdat):
            if i['type'] == 'file':
                print('f: {}/{}'.format(path.lstrip('/'), i['name']))
            else:
                print('d: {}/{}'.format(path.lstrip('/'), i['name']))

        return True

    def find(self, ep=None, path=None):

        ep = ep if ep else self._cep
        path = path if path else self._cwd
        path = '/' if path is None else path

        return self._gsm.fts('{}:/{}'.format(ep, path))

    def mv(self, ep1, path1, ep2, path2):
        if ep1 != ep2:
            return False

        return self._gsm.rename('{}:/{}'.format(ep1, path1),
                                '{}:/{}'.format(ep2, path2))

    def rm(self, ep, path):
        return self._gsm.rmdir('{}:/{}'.format(ep, path))

    def cp(self, ep1, path1, ep2, path2, recursive=False):
        return self._gsm.cp('{}:/{}'.format(ep1, path1),
                            '{}:/{}'.format(ep2, path2), recursive)

    def sh(self, prompt='globus% '):
        cmds = set(('env', 'pwd', 'cd', 'ls', 'find', 'cp', 'mv', 'rm'))

        while True:
            try:
                data = input(prompt).split(' ')
            except EOFError:
                return

            cmd, args = data[0], data[1:]

            log.debug('cmd input: {} -> ({}, {}'.format(data, cmd, args))

            if not cmd or cmd.startswith('#'):
                continue
            elif cmd in cmds:
                try:
                    args = (i.split(':') if ':' in i else (None, i)
                            for i in args)

                    ret = getattr(self, cmd)((*[j for i in args for j in i]))

                    if ret not in {True, None}:
                        print(ret)

                except Exception as e:
                    print(repr(e))
            else:
                print('commands: {}'.format(cmds))


if __name__ == '__main__':

    leps = dj.config.get('custom', {}).get('globus.storage_locations', None)
    peps = GlobusStorageLocation().fetch(as_dict=True)

    sh = GlobusShell({'local_endpoints': leps, 'publication_endpoints': peps})

    if len(sys.argv) > 1:
        sys.stdin = open(sys.argv[1], 'r')
        exit(sh.sh(prompt=''))
    else:
        exit(sh.sh())
