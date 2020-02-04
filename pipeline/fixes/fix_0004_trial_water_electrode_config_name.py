#! /usr/bin/env python

import logging

import datajoint as dj

from pipeline import lab
from pipeline import experiment
from pipeline import ephys
from pipeline import histology


log = logging.getLogger(__name__)



def add_trial_water_fields():

    conn = dj.conn()
    experiment_db = experiment.schema.database

    with conn.transaction:
        q_str = '''alter table `{}`.`_behavior_trial`
                   add `auto_water` tinyint(1) DEFAULT 0;
                '''.format(experiment_db)

        log.warning('`{}`.`_behavior_trial` auto_water'.format(experiment_db))
        res = conn.query(q_str)

        q_str = '''alter table `{}`.`_behavior_trial`
                   add `free_water` tinyint(1) DEFAULT 0;
                '''.format(experiment_db)

        log.warning('`{}`.`_behavior_trial` free_water'.format(experiment_db))
        res = conn.query(q_str)

    
def extend_electrode_config_name():

    conn = dj.conn()

    lab_db = lab.schema.database
    ephys_db = ephys.schema.database
    hist_db = histology.schema.database

    # ephys.Unit.table_name
    fixes = {
        lab_db: [lab.ElectrodeConfig, lab.ElectrodeConfig.Electrode,
                 lab.ElectrodeConfig.ElectrodeGroup],
        ephys_db: [ephys.ProbeInsertion, ephys.LFP.Channel, ephys.Unit],
        hist_db: [histology.ElectrodeCCFPosition.ElectrodePosition,
                  histology.ElectrodeCCFPosition.ElectrodePositionError,
                  histology.EphysCharacteristic]
    }

    with conn.transaction:
        for schema in [lab, ephys, histology]:
            for tbl in fixes[schema.schema.database]:

                q_str = '''
                        alter table `{}`.`{}`
                        modify `electrode_config_name` varchar(64) NOT NULL
                        comment 'user friendly name'
                        '''.format(schema.schema.database, tbl.table_name)

                log.warning('electrode_config_name `{}`.`{}`'.format(
                    schema.schema.database, tbl.table_name))
                res = conn.query(q_str)


if __name__ == '__main__':
    add_trial_water_fields()
    extend_electrode_config_name()
