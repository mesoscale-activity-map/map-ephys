from pipeline import lab, experiment, ephys
from pipeline.ingest import ephys as ephys_ingest


subjects_to_archive = ("SC011", "SC013", "SC015", "SC016", "SC017",
                       "SC022", "SC023", "SC026", "SC027", "SC030",
                       "SC031", "SC032", "SC033", "SC035", "SC038",
                       "SC043", "SC045", "SC048", "SC049", "SC050",
                       "SC052", "SC053", "SC060", "SC061", "SC064",
                       "SC065", "SC066", "SC067")


def main(limit=None):
    subject_keys = (lab.Subject * lab.WaterRestriction.proj('water_restriction_number')
                    & f'water_restriction_number in {subjects_to_archive}').fetch('KEY')

    sorted_sessions = experiment.Session & ephys.Unit - ephys.ArchivedClustering & subject_keys

    for session_key in sorted_sessions.fetch('KEY', limit=limit):
        print(f'\n----------- Archiving for {session_key} ------------')
        try:
            ephys_ingest.archive_ingested_clustering_results(session_key)
        except Exception as e:
            print(f'\tError... Skipped\n{str(e)}')


if __name__ == '__main__':
    main()
