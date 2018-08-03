
'''
Globus utilities WIP
'''

import logging

import datajoint as dj

from globus_sdk import NativeAppAuthClient
from globus_sdk import RefreshTokenAuthorizer
from globus_sdk import TransferClient
from globus_sdk import DeleteData
from globus_sdk import TransferData


log = logging.getLogger(__name__)


class GlobusStorageManager:
    # https://globus-sdk-python.readthedocs.io/en/stable/clients/transfer/

    app_id = 'b2fe5703-edb0-4f7f-80a6-2147c8ae35f0'  # map transfer app id

    class GlobusQueue:
        ''' placeholder for globus async helpers '''
        pass

    def __init__(self):

        self.auth_client = NativeAppAuthClient(self.app_id)
        self.auth_client.oauth2_start_flow(refresh_tokens=True)
        self.xfer_client = None

        if 'globus.token' not in dj.config:
            self.login()
        else:
            self.refresh()

    # authentication methods

    def login(self):
        ''' fetch refresh token, store in dj.config['globus.token'] '''

        auth_client = self.auth_client

        print('Please login via: {}'.format(
            auth_client.oauth2_get_authorize_url()))

        code = input('and enter code:').strip()
        tokens = auth_client.oauth2_exchange_code_for_tokens(code)

        xfer_auth_cfg = tokens.by_resource_server['transfer.api.globus.org']
        xfer_rt = xfer_auth_cfg['refresh_token']
        xfer_at = xfer_auth_cfg['access_token']
        xfer_exp = xfer_auth_cfg['expires_at_seconds']

        xfer_auth = RefreshTokenAuthorizer(
            xfer_rt, auth_client, access_token=xfer_at, expires_at=xfer_exp)

        self.xfer_client = TransferClient(authorizer=xfer_auth)

        dj.config['globus.token'] = xfer_rt

    def refresh(self):
        ''' use refresh token to refresh access token '''
        auth_client = self.auth_client

        xfer_auth = RefreshTokenAuthorizer(
            dj.config['globus.token'], auth_client, access_token=None,
            expires_at=None)

        self.xfer_client = TransferClient(authorizer=xfer_auth)

    # endpoint managment / utility methods

    @classmethod
    def ep_parts(cls, endpoint_path):
        # split endpoint:/path to endpoint, path
        epsplit = endpoint_path.split(':')
        return epsplit[0], ':'.join(epsplit[1:])

    def activate_endpoint(self, endpoint):
        ''' activate an endpoint '''
        tc = self.xfer_client

        r = tc.endpoint_autoactivate(endpoint, if_expires_in=3600)

        log.debug('activate_endpoint() code: {}'.format(r['code']))

        if r['code'] == 'AutoActivationFailed':
            print('Endpoint({}) Not Active! Error! Source message: {}'
                  .format(endpoint, r['message']))
            raise Exception('globus endpoint activation failure')

        knownok = any(('AutoActivated' in r['code'],
                       'AlreadyActivated' in r['code']))

        if not knownok:
            log.debug('activate_endpoint(): not knownok response')

    def _wait(self, task, timeout=10, polling_interval=10):
        ''' tranfer client common wait wrapper '''
        return self.xfer_client.task_wait(task, timeout, polling_interval)

    def _tasks(self):
        '''
        >>> tl = tc.task_list(num_results=25, filter="type:TRANSFER,DELETE")
        >>> _ = [print(t["task_id"], t["type"], t["status"]) for t in tl]
        '''
        pass

    def _task_info(self):
        '''
        >>> for event in tc.task_event_list(task_id):
        >>>     print("Event on Task({}) at {}:\n{}".format(
        >>>         task_id, event["time"], event["description"])
        or
        get_task
        '''
        pass

    # transfer methods

    def ls(self, endpoint_path):
        '''
        returns:
            {
              "DATA": [
                {
                  "DATA_TYPE": "file",
                  "group": "staff",
                  "last_modified": "2018-05-22 18:49:19+00:00",
                  "link_group": null,
                  "link_last_modified": null,
                  "link_size": null,
                  "link_target": null,
                  "link_user": null,
                  "name": "map",
                  "permissions": "0755",
                  "size": 102,
                  "type": "dir",
                  "user": "chris"
                },
              ],
              "DATA_TYPE": "file_list",
              "absolute_path": null,
              "endpoint": "aa4e5f9c-05f3-11e8-a6ad-0a448319c2f8",
              "length": 2,
              "path": "/~/Globus/",
              "rename_supported": true,
              "symlink_supported": false,
              "total": 2
            }
        '''
        ep, path = self.ep_parts(endpoint_path)
        return self.xfer_client.operation_ls(ep, path=path)

    def mkdir(self, ep_path):
        ''' create a directory at ep_path '''
        ep, path = self.ep_parts(ep_path)
        return self.xfer_client.operation_mkdir(ep, path=path)

    def rmdir(self, ep_path, recursive=False):
        ''' remove a directory at ep_path '''
        tc = self.xfer_client
        ep, path = self.ep_parts(ep_path)
        ddata = DeleteData(tc, ep, recursive=recursive)
        ddata.add_item(path)
        task_id = tc.submit_delete(ddata)['task_id']
        return self._wait(task_id)

    def cp(self, src_ep_path, dst_ep_path, recursive=False):
        '''
        copy file/path
        todo: support label, sync_level, etc?
        sync_level: ["exists", "size", "mtime", "checksum"]
        '''
        tc = self.xfer_client
        sep, spath = self.ep_parts(src_ep_path)
        dep, dpath = self.ep_parts(dst_ep_path)

        td = TransferData(tc, sep, dep)
        td.add_item(spath, dpath, recursive=recursive)

        task_id = tc.submit_transfer(td)['task_id']
        return self._wait(task_id)

    def rename(self, src_ep_path, dst_ep_path):
        ''' rename a file/path '''
        tc = self.xfer_client
        sep, spath = self.ep_parts(src_ep_path)
        dep, dpath = self.ep_parts(dst_ep_path)

        if sep != dep:
            raise Exception('rename between two different endpoints')

        return tc.operation_rename(sep, spath, dpath)
