#! /usr/bin/env python

import logging

import datajoint as dj

from pipeline import shell
from pipeline import get_schema_name

# use vmod prevent loading of new lookup data
publication = dj.create_virtual_module('publication',
                                       get_schema_name('publication'))

log = logging.getLogger(__name__)


def fix_0003_publication_filetype():

    fixmap = {
        '3a-ap-trial':                  'ephys-raw-3a-ap-trial',
        '3a-ap-trial-meta':             'ephys-raw-3a-ap-trial-meta',
        '3a-lf-trial':                  'ephys-raw-3a-lf-trial',
        '3a-lf-trial-meta':             'ephys-raw-3a-lf-trial-meta',
        '3b-ap-concat':                 'ephys-raw-3b-ap-concat',
        '3b-ap-concat-meta':            'ephys-raw-3b-ap-concat-meta',
        '3b-ap-trial':                  'ephys-raw-3b-ap-trial',
        '3b-ap-trial-meta':             'ephys-raw-3b-ap-trial-meta',
        '3b-lf-concat':                 'ephys-raw-3b-lf-concat',
        '3b-lf-concat-meta':            'ephys-raw-3b-lf-concat-meta',
        '3b-lf-trial':                  'ephys-raw-3b-lf-trial',
        '3b-lf-trial-meta':             'ephys-raw-3b-lf-trial-meta',
    }

    conn = dj.conn()
    pub_db = publication.schema.database
    q_str = '''update `{}`.`#file_type`
               set file_type='NEW'
               where file_type='ORIG';'''.format(pub_db)

    log.info('fixing publication filetype keys')

    with conn.transaction:

        for orig, new in fixmap.items():

            log.info('.. {} -> {}'.format(orig, new))
            res = conn.query(q_str.replace('ORIG', orig).replace('NEW', new))
            assert res.rowcount == 1

    log.info('ok.')


if __name__ == '__main__':
    loglevel = 'INFO'
    shell.logsetup(loglevel)
    log.setLevel(loglevel)
    fix_0003_publication_filetype()
