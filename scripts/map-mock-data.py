#! /usr/bin/env python

import os
import re
import sys
import pathlib
from importlib import reload

# pipeline_path = pathlib.Path('..').resolve()
# if pipeline_path.exists():
#     print(pipeline_path.absolute())
#     sys.path.insert(0, str(pipeline_path))

import datajoint as dj

from pipeline import lab
from pipeline import ccf
from pipeline import experiment
from pipeline import ephys
from pipeline import histology
from pipeline import tracking
from pipeline import publication
from pipeline import get_schema_name


def usage_exit():
    print("usage: {p} [{c}]"
          .format(p=os.path.basename(sys.argv[0]),
                  c='|'.join(list(actions.keys()))))
    sys.exit(0)


def dropdbs():
    print('dropping databases')
    for d in ['ingest_histology', 'ingest_ephys', 'ingest_tracking',
              'ingest_behavior', 'publication', 'psth', 'tracking',
              'histology', 'ephys', 'experiment', 'ccf', 'lab']:
        dname = get_schema_name(d)
        print('..  {} ({})'.format(d, dname))
        try:
            schema = dj.schema(dname)
            schema.drop(force=True)
        except Exception as e:
            print("....  couldn't drop database {} : {}".format(d, repr(e)))


def mockdata():
    print('populating with mock data')
    reload(lab)
    reload(ccf)
    reload(experiment)
    reload(ephys)
    reload(histology)
    reload(tracking)
    reload(publication)
    try:
        # TODO: these should be loaded in a more 'official' way
        lab.Person().insert1({
            'username': 'unknown',
            'fullname': 'Unknown'},
            skip_duplicates=True
        )
        lab.Person().insert1({
            'username': 'daveliu',
            'fullname': 'Dave Liu'},
            skip_duplicates=True
        )
        lab.Person().insert1({
            'username': 'susu',
            'fullname': 'Susu Chen'},
            skip_duplicates=True
        )
        lab.ModifiedGene().insert1({
            'gene_modification': 'VGAT-Chr2-EYFP Jax',
            'gene_modification_description': 'VGAT'},
            skip_duplicates=True
        )
        lab.ModifiedGene().insert1({
            'gene_modification': 'PV-ires-Cre X Ai32',
            'gene_modification_description': 'PV'},
            skip_duplicates=True
        )
        lab.ModifiedGene().insert1({
            'gene_modification': 'Rosa26 Cag lsl reachR-citrine 1A4 X PV-ires-Cre',
            'gene_modification_description': 'reachR PV'},
            skip_duplicates=True
        )
        # Subject 399752 / dl7
        lab.Subject().insert1({
            'subject_id': 399752,
            'username': 'daveliu',
            'cage_number': 145375,
            'date_of_birth': '2017-08-03',
            'sex': 'M',
            'animal_source': 'Jackson labs'},
            skip_duplicates=True
        )
        lab.Subject.GeneModification().insert1({
            'subject_id': 399752,
            'gene_modification': 'VGAT-Chr2-EYFP Jax'},
            skip_duplicates=True
        )
        lab.Surgery().insert1({
            'subject_id': 399752,
            'surgery_id': 1,
            'username': 'daveliu',
            'start_time': '2017-11-03',
            'end_time': '2017-11-03',
            'surgery_description': 'Headbar anterior'},
            skip_duplicates=True
        )
        lab.Surgery.Procedure().insert1({
            'subject_id': 399752,
            'surgery_id': 1,
            'procedure_id': 1,
            'skull_reference': 'Bregma',
            'ml_location': 0,
            'ap_location': -4,
            'surgery_procedure_description': 'Fiducial marker'},
            skip_duplicates=True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 399752,
            'water_restriction_number': 'dl7',
            'cage_number': 148861,
            'wr_start_date': '2017-11-07',
            'wr_start_weight': 25},
            skip_duplicates=True
        )
        # Subject 397853 / dl14
        lab.Subject().insert1({
            'subject_id': 397853,
            'username': 'daveliu',
            'cage_number': 144545,
            'date_of_birth': '2017-07-15',
            'sex':  'M',
            'animal_source': 'Allen Institute'},
            skip_duplicates=True
        )
        lab.Subject.GeneModification().insert1({
            'subject_id': 397853,
            'gene_modification': 'PV-ires-Cre X Ai32'},
            skip_duplicates=True
        )
        lab.Surgery().insert1({
            'subject_id': 397853,
            'surgery_id': 1,
            'username': 'daveliu',
            'start_time': '2017-11-20',
            'end_time': '2017-11-20',
            'surgery_description': 'Headbar posterior'},
            skip_duplicates=True
        )
        lab.Surgery.Procedure().insert1({
            'subject_id': 397853,
            'surgery_id': 1,
            'procedure_id': 1,
            'skull_reference': 'Bregma',
            'ml_location': 0,
            'ap_location': -1.75,
            'surgery_procedure_description': 'Fiducial marker'},
            skip_duplicates=True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 397853,
            'water_restriction_number': 'dl14',
            'cage_number': 149595,
            'wr_start_date': '2017-11-27',
            'wr_start_weight': 24.1},
            skip_duplicates=True
        )
        # Subject 400480 / dl15
        lab.Subject().insert1({
            'subject_id': 400480,
            'username': 'daveliu',
            'cage_number': 145700,
            'date_of_birth': '2017-08-09',
            'sex':  'M',
            'animal_source': 'Allen Institute'},
            skip_duplicates=True
        )
        lab.Subject.GeneModification().insert1({
            'subject_id': 400480,
            'gene_modification': 'PV-ires-Cre X Ai32'},
            skip_duplicates=True
        )
        lab.Surgery().insert1({
            'subject_id': 400480,
            'surgery_id': 1,
            'username': 'daveliu',
            'start_time': '2017-11-21',
            'end_time': '2017-11-21',
            'surgery_description': 'Headbar posterior'},
            skip_duplicates=True
        )
        lab.Surgery.Procedure().insert1({
            'subject_id': 400480,
            'surgery_id': 1,
            'procedure_id': 1,
            'skull_reference': 'Bregma',
            'ml_location': 0,
            'ap_location': -1.75,
            'surgery_procedure_description': 'Fiducial marker'},
            skip_duplicates=True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 400480,
            'water_restriction_number': 'dl15',
            'cage_number': 149598,
            'wr_start_date': '2017-11-27',
            'wr_start_weight': 27.6},
            skip_duplicates=True
        )
        # Subject 406680 / dl20
        lab.Subject().insert1({
            'subject_id': 406680,
            'username': 'daveliu',
            'cage_number': 148859,
            'date_of_birth': '2017-10-06',
            'sex':  'F',
            'animal_source': 'Jackson labs'},
            skip_duplicates=True
        )
        lab.Subject.GeneModification().insert1({
            'subject_id': 406680,
            'gene_modification': 'VGAT-Chr2-EYFP Jax'},
            skip_duplicates=True
        )
        lab.Surgery().insert1({
            'subject_id': 406680,
            'surgery_id': 1,
            'username': 'daveliu',
            'start_time': '2018-01-04',
            'end_time': '2018-01-04',
            'surgery_description': 'Headbar posterior'},
            skip_duplicates=True
        )
        lab.Surgery.Procedure().insert1({
            'subject_id': 406680,
            'surgery_id': 1,
            'procedure_id': 1,
            'skull_reference': 'Bregma',
            'ml_location': 0,
            'ap_location': -1.75,
            'surgery_procedure_description': 'Fiducial marker'},
            skip_duplicates=True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 406680,
            'water_restriction_number': 'dl20',
            'cage_number': 151282,
            'wr_start_date': '2018-01-10',
            'wr_start_weight': 22.7},
            skip_duplicates=True
        )
        # Subject 408022 / dl21
        lab.Subject().insert1({
            'subject_id': 408022,
            'username': 'daveliu',
            'cage_number': 148859,
            'date_of_birth': '2017-10-19',
            'sex':  'F',
            'animal_source': 'Jackson labs'},
            skip_duplicates=True
        )
        lab.Subject.GeneModification().insert1({
            'subject_id': 408022,
            'gene_modification': 'VGAT-Chr2-EYFP Jax'},
            skip_duplicates=True
        )
        lab.Surgery().insert1({
            'subject_id': 408022,
            'surgery_id': 1,
            'username': 'daveliu',
            'start_time': '2018-01-05',
            'end_time': '2018-01-05',
            'surgery_description': 'Headbar posterior'},
            skip_duplicates=True
        )
        lab.Surgery.Procedure().insert1({
            'subject_id': 408022,
            'surgery_id': 1,
            'procedure_id': 1,
            'skull_reference': 'Bregma',
            'ml_location': 0,
            'ap_location': -1.75,
            'surgery_procedure_description': 'Fiducial marker'},
            skip_duplicates=True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 408022,
            'water_restriction_number': 'dl21',
            'cage_number': 151283,
            'wr_start_date': '2018-01-10',
            'wr_start_weight': 21.1},
            skip_duplicates=True
        )
        # Subject 408021 / dl22
        lab.Subject().insert1({
            'subject_id': 408021,
            'username': 'daveliu',
            'cage_number': 148859,
            'date_of_birth': '2017-10-19',
            'sex':  'F',
            'animal_source': 'Jackson labs'},
            skip_duplicates=True
        )
        lab.Subject.GeneModification().insert1({
            'subject_id': 408021,
            'gene_modification': 'VGAT-Chr2-EYFP Jax'},
            skip_duplicates=True
        )
        lab.Surgery().insert1({
            'subject_id': 408021,
            'surgery_id': 1,
            'username': 'daveliu',
            'start_time': '2018-01-15',
            'end_time': '2018-01-15',
            'surgery_description': 'Headbar posterior'},
            skip_duplicates=True
        )
        lab.Surgery.Procedure().insert1({
            'subject_id': 408021,
            'surgery_id': 1,
            'procedure_id': 1,
            'skull_reference': 'Bregma',
            'ml_location': 0,
            'ap_location': -1.75,
            'surgery_procedure_description': 'Fiducial marker'},
            skip_duplicates=True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 408021,
            'water_restriction_number': 'dl22',
            'cage_number': 151704,
            'wr_start_date': '2018-01-19',
            'wr_start_weight': 21},
            skip_duplicates=True
        )
        # Subject 407512 / dl24
        lab.Subject().insert1({
            'subject_id': 407512,
            'username': 'daveliu',
            'cage_number': 151629,
            'date_of_birth': '2017-10-13',
            'sex':  'M',
            'animal_source': 'Jackson labs'},
            skip_duplicates=True
        )
        lab.Subject.GeneModification().insert1({
            'subject_id': 407512,
            'gene_modification': 'VGAT-Chr2-EYFP Jax'},
            skip_duplicates=True
        )
        lab.Surgery().insert1({
            'subject_id': 407512,
            'surgery_id': 1,
            'username': 'daveliu',
            'start_time': '2018-01-16',
            'end_time': '2018-01-16',
            'surgery_description': 'Headbar posterior'},
            skip_duplicates=True
        )
        lab.Surgery.Procedure().insert1({
            'subject_id': 407512,
            'surgery_id': 1,
            'procedure_id': 1,
            'skull_reference': 'Bregma',
            'ml_location': 0,
            'ap_location': -1.75,
            'surgery_procedure_description': 'Fiducial marker'},
            skip_duplicates=True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 407512,
            'water_restriction_number': 'dl24',
            'cage_number': 151793,
            'wr_start_date': '2018-01-22',
            'wr_start_weight': 26},
            skip_duplicates=True
        )
        # 407513 / dl25
        lab.Subject().insert1({
            'subject_id': 407513,
            'username': 'daveliu',
            'cage_number': 148636,
            'date_of_birth': '2017-10-13',
            'sex':  'M',
            'animal_source': 'Jackson labs'},
            skip_duplicates=True
        )
        lab.Subject.GeneModification().insert1({
            'subject_id': 407513,
            'gene_modification': 'VGAT-Chr2-EYFP Jax'},
            skip_duplicates=True
        )
        lab.Surgery().insert1({
            'subject_id': 407513,
            'surgery_id': 1,
            'username': 'daveliu',
            'start_time': '2018-01-17',
            'end_time': '2018-01-17',
            'surgery_description': 'Headbar posterior'},
            skip_duplicates=True
        )
        lab.Surgery.Procedure().insert1({
            'subject_id': 407513,
            'surgery_id': 1,
            'procedure_id': 1,
            'skull_reference': 'Bregma',
            'ml_location': 0,
            'ap_location': -1.75,
            'surgery_procedure_description': 'Fiducial marker'},
            skip_duplicates=True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 407513,
            'water_restriction_number': 'dl25',
            'cage_number': 151794,
            'wr_start_date': '2018-01-22',
            'wr_start_weight': 25.5},
            skip_duplicates=True
        )
        # Subject 407986 / dl28
        lab.Subject().insert1({
            'subject_id': 407986,
            'username': 'daveliu',
            'cage_number': 152268,
            'date_of_birth': '2017-10-18',
            'sex':  'F',
            'animal_source': 'Jackson labs'},
            skip_duplicates=True
        )
        lab.Subject.GeneModification().insert1({
            'subject_id': 407986,
            'gene_modification': 'VGAT-Chr2-EYFP Jax'},
            skip_duplicates=True
        )
        lab.Surgery().insert1({
            'subject_id': 407986,
            'surgery_id': 1,
            'username': 'daveliu',
            'start_time': '2018-02-01',
            'end_time': '2018-02-01',
            'surgery_description': 'Headbar anterior'},
            skip_duplicates=True
        )
        lab.Surgery.Procedure().insert1({
            'subject_id': 407986,
            'surgery_id': 1,
            'procedure_id': 1,
            'skull_reference': 'Bregma',
            'ml_location': 0,
            'ap_location': -4,
            'surgery_procedure_description': 'Fiducial marker'},
            skip_duplicates=True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 407986,
            'water_restriction_number': 'dl28',
            'cage_number': 152312,
            'wr_start_date': '2018-02-05',
            'wr_start_weight': 19.8},
            skip_duplicates=True
        )
        # Subject 123457 / tw5
        lab.Subject().insert1({
            'subject_id': 123457,
            'username': 'daveliu',
            'cage_number': 145375,
            'date_of_birth': '2017-08-03',
            'sex': 'M',
            'animal_source': 'Jackson labs'},
            skip_duplicates=True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 123457,
            'water_restriction_number': 'tw5',
            'cage_number': 148861,
            'wr_start_date': '2017-11-07',
            'wr_start_weight': 20.5},
            skip_duplicates=True
        )
        # Subject 412330 / tw34
        lab.Subject().insert1({
            'subject_id': 412330,
            'username': 'daveliu',
            'cage_number': 154522,
            'date_of_birth': '2017-12-05',
            'sex': 'M',
            'animal_source': 'Jackson labs'},
            skip_duplicates=True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 412330,
            'water_restriction_number': 'tw34',
            'cage_number': 154522,
            'wr_start_date': '2018-03-18',
            'wr_start_weight': 21.0},
            skip_duplicates=True
        )
        # subject 432998 / dl55
        lab.Subject().insert1({
            'subject_id': 432998,
            'username': 'daveliu',
            'cage_number': 160920,
            'date_of_birth': '2018-07-02',
            'sex': 'M',
            'animal_source': 'Jackson labs'},
            skip_duplicates=True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 432998,
            'water_restriction_number': 'dl55',
            'cage_number': 160920,
            'wr_start_date': '2018-09-05',
            'wr_start_weight': 21.0},
            skip_duplicates=True
        )
        # Subject 435884 / dl59
        lab.Subject().insert1({
            'subject_id': 435884,
            'username': 'daveliu',
            'cage_number': 161908,
            'date_of_birth': '2018-08-06',
            'sex': 'M',
            'animal_source': 'Jackson labs'},
            skip_duplicates=True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 435884,
            'water_restriction_number': 'dl59',
            'cage_number': 154522,
            'wr_start_date': '2018-09-30',
            'wr_start_weight': 21.0},
            skip_duplicates=True
        )
        # Subject 432572 / dl56
        lab.Subject().insert1({
            'subject_id': 432572,
            'username': 'daveliu',
            'cage_number': 161125,
            'date_of_birth': '2018-06-28',
            'sex': 'M',
            'animal_source': 'Jackson labs'},
            skip_duplicates=True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 432572,
            'water_restriction_number': 'dl56',
            'cage_number': 161125,
            'wr_start_date': '2018-09-10',
            'wr_start_weight': 21.0},
            skip_duplicates=True
        )

        # Subject 412753 / dl36
        lab.Subject().insert1({
            'subject_id': 412753,
            'username': 'daveliu',
            'cage_number': 154570,
            'date_of_birth': '2017-12-07',
            'sex': 'M',
            'animal_source': 'Jackson labs'},
            skip_duplicates=True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 412753,
            'water_restriction_number': 'dl36',
            'cage_number': 154570,
            'wr_start_date': '2017-03-30',
            'wr_start_weight': 21.0},
            skip_duplicates=True
        )

        # Subject 888888 / SC022
        lab.Subject().insert1({
            'subject_id': 888888,
            'username': 'susu',
            'cage_number': 888888,
            'date_of_birth': '2018-08-06',
            'sex': 'M',
            'animal_source': 'Jackson labs'},
            skip_duplicates=True
        )
        lab.WaterRestriction().insert1({
            'subject_id': 888888,
            'water_restriction_number': 'SC022',
            'cage_number': 888888,
            'wr_start_date': '2018-09-30',
            'wr_start_weight': 21.0},
            skip_duplicates=True
        )
        # Rig
        lab.Rig().insert1({
            'rig': 'TRig1',
            'room': '2w.334',
            'rig_description': 'Training rig 1'},
            skip_duplicates=True
        )
        lab.Rig().insert1({
            'rig': 'TRig2',
            'room': '2w.334',
            'rig_description': 'Training rig 2'},
            skip_duplicates=True
        )
        lab.Rig().insert1({
            'rig': 'TRig3',
            'room': '2w.334',
            'rig_description': 'Training rig 3'},
            skip_duplicates=True
        )
        lab.Rig().insert1({
            'rig': 'RRig',
            'room': '2w.334',
            'rig_description': 'Recording rig'},
            skip_duplicates=True
        )
        lab.Rig().insert1({
            'rig': 'RRig2',
            'room': '2w.334',
            'rig_description': 'Recording rig2'},
            skip_duplicates=True
        )
        lab.Rig().insert1({
            'rig': 'Ephys1',
            'room': '2w.334',
            'rig_description': 'Recording computer'},
            skip_duplicates=True
        )

        # ---- Brain Location ----
        experiment.BrainLocation.insert1({
            'brain_location_name': 'left_alm',
            'brain_area': 'ALM',
            'hemisphere': 'left',
            'skull_reference': 'Bregma'},
            skip_duplicates=True
        )

        experiment.BrainLocation.insert1({
            'brain_location_name': 'right_alm',
            'brain_area': 'ALM',
            'hemisphere': 'right',
            'skull_reference': 'Bregma'},
            skip_duplicates=True
        )

        experiment.BrainLocation.insert1({
            'brain_location_name': 'both_alm',
            'brain_area': 'ALM',
            'hemisphere': 'both',
            'skull_reference': 'Bregma'},
            skip_duplicates=True
        )

        experiment.BrainLocation.insert1({
            'brain_location_name': 'left_medulla',
            'brain_area': 'Medulla',
            'hemisphere': 'left',
            'skull_reference': 'Bregma'},
            skip_duplicates=True
        )

        experiment.BrainLocation.insert1({
            'brain_location_name': 'right_medulla',
            'brain_area': 'Medulla',
            'hemisphere': 'right',
            'skull_reference': 'Bregma'},
            skip_duplicates=True
        )

        experiment.BrainLocation.insert1({
            'brain_location_name': 'both_medulla',
            'brain_area': 'Medulla',
            'hemisphere': 'both',
            'skull_reference': 'Bregma'},
            skip_duplicates=True
        )

        experiment.BrainLocation.insert1({
            'brain_location_name': 'left_striatum',
            'brain_area': 'Striatum',
            'hemisphere': 'left',
            'skull_reference': 'Bregma'},
            skip_duplicates=True
        )

        experiment.BrainLocation.insert1({
            'brain_location_name': 'right_striatum',
            'brain_area': 'Striatum',
            'hemisphere': 'right',
            'skull_reference': 'Bregma'},
            skip_duplicates=True
        )

        experiment.BrainLocation.insert1({
            'brain_location_name': 'both_striatum',
            'brain_area': 'Striatum',
            'hemisphere': 'both',
            'skull_reference': 'Bregma'},
            skip_duplicates=True
        )

        experiment.BrainLocation.insert1({
            'brain_location_name': 'left_thalamus',
            'brain_area': 'Thalamus',
            'hemisphere': 'left',
            'skull_reference': 'Bregma'},
            skip_duplicates=True
        )
        experiment.BrainLocation.insert1({
            'brain_location_name': 'right_thalamus',
            'brain_area': 'Thalamus',
            'hemisphere': 'right',
            'skull_reference': 'Bregma'},
            skip_duplicates=True
        )
        experiment.BrainLocation.insert1({
            'brain_location_name': 'both_thalamus',
            'brain_area': 'Thalamus',
            'hemisphere': 'both',
            'skull_reference': 'Bregma'},
            skip_duplicates=True
        )

        # Probe (Neuropixel)
        npx_probe_model = '15131808323'   # using Model No. - SN TBD?
        lab.Probe.insert1({
            'probe': npx_probe_model,
            'probe_type': 'neuropixel'},
            skip_duplicates=True,
        )
        lab.Probe.Electrode.insert(
            ({'probe': npx_probe_model, 'electrode': x} for
             x in range(1, 961)),
            skip_duplicates=True,
        )

    except Exception as e:
        print("error creating mock data: {e}".format(e=e), file=sys.stderr)
        raise


def post_ephys(*args):
    from pipeline.ingest import ephys as ephys_ingest
    for ef in ephys_ingest.EphysIngest.EphysFile().fetch(as_dict=True):
        fname = ef['ephys_file']
        print('attempting Probe InsertionLocation for fname: ', end='')
        if re.match('.*2018-12-07.*dl59.*.mat', fname):
            recs = [{
                'subject_id': 435884,
                'session': 1,
                'insertion_number': 1,
                'brain_location_name': 'right_alm',
                'ml_location': 1500,
                'ap_location': 2500,
                'dv_location': 1668.9,
                # ml_angle:
                # ap_angle:
            }]
            print('match!: {}'.format(recs))
        elif re.match('.*2018-07-16.*tw34.*.mat', fname):
            recs = [{
                'subject_id': 412330,
                'session': 1,
                'insertion_number': 1,
                'brain_location_name': 'right_alm',
                'ml_location': 1500,
                'ap_location': 2500,
                'dv_location': 2103.2,
                # ml_angle:
                # ap_angle:
            },{
                'subject_id': 412330,
                'session': 1,
                'insertion_number': 2,
                'brain_location_name': 'right_medulla',
                'ml_location': 1000,
                'ap_location': 6500,
                'dv_location': 5237.5,
                # ml_angle:
                # ap_angle:
            }]
            print('match!: {}'.format(recs))
        elif re.match('.*SC022/2019-03-03.*', fname):
            # FIXME: 15deg dv angle -> ?
            # FIXME: 'you should add 175um to the note depths
            #   'because where I used as zero reference is where the 1st tip
            #    enters the dura' ...
            kbase = {'subject_id': 888888, 'session': 1}
            recs = [{
                **kbase,
                'insertion_number': 1,
                'brain_location_name': 'left_alm',
                'ml_location': 1500,
                'ap_location': 2500,
                'dv_location': 2900,
                # ml_angle:
                # ap_angle:
            },{
                **kbase,
                'insertion_number': 2,
                'brain_location_name': 'right_alm',
                'ml_location': 1500,
                'ap_location': 2500,
                'dv_location': 2900,
                # ml_angle:
                # ap_angle:
            },{
                **kbase,
                'insertion_number': 3,
                'brain_location_name': 'left_striatum',
                'ml_location': 1000,
                'ap_location': 2800,
                'dv_location': 4500,
                # ml_angle:
                # ap_angle:
            },{
                **kbase,
                'insertion_number': 4,
                'brain_location_name': 'left_striatum',
                'ml_location': 1000,
                'ap_location': 2800,
                'dv_location': 4500,
                # ml_angle:
                # ap_angle:
            }]
            print('match!: {}'.format(recs))
        elif re.match('.*2018-07-10.*dl36.*.mat', fname):
            kbase = {'subject_id': 412753, 'session': 1}
            recs = [{
                **kbase,
                'insertion_number': 1,
                'brain_location_name': 'right_alm',
                'ml_location': 1000,
                'ap_location': 2500,
                'dv_location': 2019.4,
                # ml_angle:
                # ap_angle:
            },{
                **kbase,
                'insertion_number': 2,
                'brain_location_name': 'right_thalamus',
                'ml_location': 1000,
                'ap_location': 1750,
                'dv_location': 4954.8,
                # ml_angle:
                # ap_angle:
            }]
            print('match!: {}'.format(recs))
        else:
            print('no match!')

        ephys.ProbeInsertion.InsertionLocation().insert(
            recs, skip_duplicates=True)
        ephys.ProbeInsertion.RecordingSystemSetup().insert(
            [{**r, 'sampling_rate': 30000} for r in recs],
            ignore_extra_fields=True, skip_duplicates=True)


def preload(*args):
    dropdbs()
    mockdata()


actions = {
    'preload': preload,
    'post-ephys': post_ephys,
}


if __name__ == '__main__':

    if len(sys.argv) < 2 or sys.argv[1] not in actions:
        usage_exit()

    action = sys.argv[1]
    actions[action](sys.argv[2:])
