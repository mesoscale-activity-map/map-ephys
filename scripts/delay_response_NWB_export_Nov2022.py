import os
import pathlib

from pipeline import experiment
from pipeline.export.nwb import export_recording


output_dir = pathlib.Path(r'D:/map/NWB_EXPORT/delay_response')

project_name = 'Brain-wide neural activity underlying memory-guided movement'


def export_to_nwb(limit=None):
    session_keys = (experiment.Session & (experiment.ProjectSession
                                          & {'project_name': project_name})).fetch(
        'KEY', limit=limit)
    export_recording(session_keys, output_dir=output_dir, overwrite=False, validate=False)


dandiset_id = os.getenv('DANDISET_ID')
dandi_api_key = os.getenv('DANDI_API_KEY')


def publish_to_dandi(dandiset_id, dandi_api_key):
    from element_interface.dandi import upload_to_dandi

    dandiset_dir = output_dir / 'dandi'
    dandiset_dir.mkdir(parents=True, exist_ok=True)

    upload_to_dandi(
        data_directory=output_dir,
        dandiset_id=dandiset_id,
        staging=False,
        working_directory=dandiset_dir,
        api_key=dandi_api_key,
        sync=True,
        existing='overwrite')
