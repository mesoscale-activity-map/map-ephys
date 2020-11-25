#! /usr/bin/env python

import dataojoint as dj

from pipeline.histology import ElectrodeCCFPosition


def adjust_electrode_ccf_position_tables():

    tables = [ElectrodeCCFPosition.ElectrodePosition,
              ElectrodeCCFPosition.ElectrodePositionError]

    pk_new = ('PRIMARY KEY (`subject_id`,`session`,`insertion_number`,'
              '`probe_type`,`electrode_config_name`,`electrode_group`,'
              '`electrode`,`ccf_label_id`,`ccf_x`,`ccf_y`,`ccf_z`)')

    pk_alter_cmd = 'alter table {} drop primary key, add {}'

    mri_alter_cmd = ("alter table {} modify column `mri_{}` "
                     "float default NULL comment '(mm)'")

    with dj.conn().transaction as conn:

        for table in tables:

            print('altering {}'.format(table))

            print('altering primary key')
            tname = table.full_table_name
            conn.query(pk_alter_cmd.format(tname, pk_new))

            for mri in ('x', 'y', 'z'):
                print('altering mri_{}'.format(mri))
                conn.query(mri_alter_cmd.format(tname, mri))


if __name__ == '__main__':
    adjust_electrode_ccf_position_tables()
