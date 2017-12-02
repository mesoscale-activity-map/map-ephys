#! /usr/bin/env python

import os
import sys
import logging

import scipy.io as spio

import datajoint as dj

import lab
import experiment


if 'imported_session_path' not in dj.config:
    dj.config['imported_session_path'] = 'R:\\Arduino\\Bpod_Train1\\Bpod Local\\Data\\dl7\\TW_autoTrain\\Session Data\\'

log = logging.getLogger(__name__)
schema = dj.schema(dj.config['ingest.database'], locals())


def _listfiles():
    return (f for f in os.listdir(dj.config['imported_session_path'])
            if f.endswith('.mat'))


@schema
class ImportedSessionFile(dj.Lookup):
    # TODO: more representative class name
    definition = """
    imported_session_file:         varchar(255)    # imported session file
    """

    contents = ((f,) for f in (_listfiles()))

    def populate(self):
        for f in _listfiles():
            if not self & {'imported_session_file': f}:
                self.insert1((f,))


@schema
class ImportedSessionFileIngest(dj.Imported):
    definition = """
    -> ImportedSessionFile
    ---
    -> experiment.Session
    """

    def make(self, key):

        fname = key['imported_session_file']
        fpath = os.path.join(dj.config['imported_session_path'], fname)

        log.info('ImportedSessionFileIngest.make(): Loading {f}'
                 .format(f=fname))

        # split files like 'dl7_TW_autoTrain_20171114_140357.mat'
        h2o, t1, t2, date, time = fname.split('.')[0].split('_')
        
        if os.stat(fpath).st_size/1024 > 500: #False:  # TODO: pre-populate lab.Animal and AnimalWaterRestriction

            # '%%' due to datajoint-python/issues/376
            #dups = (self & "imported_session_file like '%%{h2o}%%{date}%%"
                    #.format(h2o=h2o, date=date))

            #if len(dups) > 1:
                #log.warning('split session case detected')
                # TODO: handle split file
                # TODO: self.insert( all split files )
                #return

            # lookup animal

            key['animal'] = (lab.Animal()
                             & (lab.AnimalWaterRestriction
                                and {'water_restriction': h2o})).fetch1('animal')

            # synthesize session
            key['session'] = (dj.U().aggr(experiment.Session(),
                                          n='max(session)').fetch1('n') or 0)+1

            #if experiment.Session() & key:
                # XXX: raise DataJointError?
                #log.warning("Warning! session exists for {f}".format(fname))

            mat = spio.loadmat(fpath, squeeze_me=True)  # NOQA
            # ... do rest of data loading here
            # ... and save a record here to prevent future loading
            print(key)
            self.insert1(key, ignore_extra_fields=True)


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] != 'populate':
        print("usage: {p} [populate]"
              .format(p=os.path.basename(sys.argv[0])))
        sys.exit(0)

    logging.basicConfig(level=logging.ERROR)  # quiet other modules
    log.setLevel(logging.INFO)  # but show ours
    ImportedSessionFile().populate()
    ImportedSessionFileIngest().populate()
