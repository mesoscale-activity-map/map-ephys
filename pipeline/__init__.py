
import logging

import datajoint as dj


log = logging.getLogger(__name__)


class InsertBuffer(object):
    '''
    InsertBuffer: a utility class to help managed chunked inserts

    Currently requires records do not have prerequisites.
    '''
    def __init__(self, rel):
        self._rel = rel
        self._queue = []

    def insert1(self, r):
        self._queue.append(r)

    def insert(self, recs):
        self._queue += recs

    def flush(self, replace=False, skip_duplicates=False,
              ignore_extra_fields=False, ignore_errors=False,
              allow_direct_insert=False, chunksz=1):
        '''
        flush the buffer
        XXX: use kwargs?
        XXX: ignore_extra_fields na, requires .insert() support
        '''
        qlen = len(self._queue)
        if qlen > 0 and qlen % chunksz == 0:
            try:
                self._rel.insert(self._queue, skip_duplicates=skip_duplicates,
                                 ignore_extra_fields=ignore_extra_fields,
                                 ignore_errors=ignore_errors,
                                 allow_direct_insert=allow_direct_insert)
                self._queue.clear()
                return True
            except dj.DataJointError as e:
                log.error('error in flush: {}'.format(e))
                raise
