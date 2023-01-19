import os
import datajoint as dj
import pathlib

from pipeline import lab, experiment, ephys
from pipeline.export.nwb import export_recording


output_dir = pathlib.Path(r'D:/map/NWB_EXPORT/delay_response')

subjects_to_export = ("SC011", "SC013", "SC015", "SC016", "SC017",
                      "SC022", "SC023", "SC026", "SC027", "SC030",
                      "SC031", "SC032", "SC033", "SC035", "SC038",
                      "SC043", "SC045", "SC048", "SC049", "SC050",
                      "SC052", "SC053", "SC060", "SC061", "SC064",
                      "SC065", "SC066", "SC067")


def main(limit=None):
    subject_keys = (lab.Subject * lab.WaterRestriction.proj('water_restriction_number')
                    & f'water_restriction_number in {subjects_to_export}').fetch('KEY')
    session_keys = (experiment.Session & ephys.Unit & subject_keys).fetch('KEY', limit=limit)
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
